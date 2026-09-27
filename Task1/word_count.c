#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>

#define BUCKETS 4096
typedef struct Entry { char *word; size_t count; struct Entry *next; } Entry;
typedef struct { Entry *buckets[BUCKETS]; } Table;
typedef struct {
    const char *text;
    size_t length, next, chunk;
    pthread_mutex_t lock;
} Work;
typedef struct { Work *work; Table table; size_t words, slices; int failed; } Worker;

static int word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
static unsigned hash_word(const char *s) {
    unsigned h = 5381;
    while (*s) h = h * 33u + (unsigned char)*s++;
    return h % BUCKETS;
}
static int add_word(Table *table, const char *word, size_t count) {
    unsigned h = hash_word(word);
    for (Entry *e = table->buckets[h]; e; e = e->next) {
        if (strcmp(e->word, word) == 0) { e->count += count; return 1; }
    }
    Entry *e = malloc(sizeof(*e));
    if (!e) return 0;
    e->word = strdup(word);
    if (!e->word) { free(e); return 0; }
    e->count = count; e->next = table->buckets[h]; table->buckets[h] = e;
    return 1;
}
static void free_table(Table *t) {
    for (unsigned i = 0; i < BUCKETS; ++i) {
        Entry *e = t->buckets[i];
        while (e) { Entry *next = e->next; free(e->word); free(e); e = next; }
    }
}
static void *count_words(void *arg) {
    Worker *worker = arg;
    Work *w = worker->work;
    for (;;) {
        // Only claiming a slice is shared. Counting stays in this thread's table.
        pthread_mutex_lock(&w->lock);
        size_t start = w->next;
        size_t end = w->length - start < w->chunk ? w->length : start + w->chunk;
        w->next = end;
        pthread_mutex_unlock(&w->lock);
        if (start == w->length) break;
        worker->slices++;
        // The thread owning a word's first byte also counts its remaining bytes.
        if (start && word_char((unsigned char)w->text[start-1]))
            while (start < w->length && word_char((unsigned char)w->text[start])) start++;
        while (start < end) {
            while (start < end && !word_char((unsigned char)w->text[start])) start++;
            if (start == end) break;
            size_t stop = start;
            while (stop < w->length && word_char((unsigned char)w->text[stop])) stop++;
            char *word = malloc(stop - start + 1);
            if (!word) { worker->failed = 1; return NULL; }
            for (size_t j = start; j < stop; ++j) {
                unsigned char c = (unsigned char)w->text[j];
                word[j-start] = (char)(c >= 'A' && c <= 'Z' ? c + ('a'-'A') : c);
            }
            word[stop-start] = '\0';
            int ok = add_word(&worker->table, word, 1);
            free(word);
            if (!ok) { worker->failed = 1; return NULL; }
            worker->words++; start = stop;
        }
    }
    return NULL;
}
static int compare_entries(const void *a, const void *b) {
    return strcmp((*(Entry *const *)a)->word, (*(Entry *const *)b)->word);
}
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s input.txt threads\n", argv[0]); return 1; }
    char *tail; errno = 0; long requested = strtol(argv[2], &tail, 10);
    if (errno || *tail || tail == argv[2] || requested < 1 || requested > 4096) {
        fprintf(stderr, "Thread count must be an integer from 1 to 4096.\n"); return 1;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) { perror(argv[1]); return 1; }
    if (fseek(input, 0, SEEK_END)) { fclose(input); return 1; }
    long bytes = ftell(input);
    if (bytes < 0 || (uintmax_t)bytes >= SIZE_MAX || fseek(input, 0, SEEK_SET)) {
        fprintf(stderr, "Cannot determine file size.\n"); fclose(input); return 1;
    }
    char *text = malloc((size_t)bytes + 1);
    if (!text) { fclose(input); return 1; }
    size_t read = fread(text, 1, (size_t)bytes, input);
    int bad_read = ferror(input) || read != (size_t)bytes;
    fclose(input);
    if (bad_read) { fprintf(stderr, "Could not read complete file.\n"); free(text); return 1; }
    text[read] = '\0';
    size_t n = (size_t)requested;
    if (read && n > read) n = read;
    if (!read) n = 1;
    Work work = { .text = text, .length = read, .chunk = read / (n * 8) + 1 };
    if (pthread_mutex_init(&work.lock, NULL)) { free(text); return 1; }
    Worker *workers = calloc(n, sizeof(*workers));
    pthread_t *threads = malloc(n * sizeof(*threads));
    Table merged = {0}; Entry **sorted = NULL;
    int status = 1; size_t made = 0, unique = 0, total = 0;
    double started = seconds();
    if (!workers || !threads) goto cleanup;
    for (; made < n; ++made) {
        workers[made].work = &work;
        int rc = pthread_create(&threads[made], NULL, count_words, &workers[made]);
        if (rc) { fprintf(stderr, "pthread_create: %s\n", strerror(rc)); break; }
    }
    for (size_t i = 0; i < made; ++i) pthread_join(threads[i], NULL);
    if (made != n) goto cleanup;
    for (size_t i = 0; i < n; ++i) {
        if (workers[i].failed) { fprintf(stderr, "Not enough memory while counting.\n"); goto cleanup; }
        total += workers[i].words;
        for (unsigned b = 0; b < BUCKETS; ++b)
            for (Entry *e = workers[i].table.buckets[b]; e; e = e->next)
                if (!add_word(&merged, e->word, e->count)) goto cleanup;
    }
    for (unsigned b = 0; b < BUCKETS; ++b)
        for (Entry *e = merged.buckets[b]; e; e = e->next) unique++;
    sorted = malloc((unique ? unique : 1) * sizeof(*sorted));
    if (!sorted) goto cleanup;
    size_t k = 0;
    for (unsigned b = 0; b < BUCKETS; ++b)
        for (Entry *e = merged.buckets[b]; e; e = e->next) sorted[k++] = e;
    qsort(sorted, unique, sizeof(*sorted), compare_entries);
    double elapsed = seconds() - started;
    FILE *output = fopen("result.txt", "w");
    if (!output) { perror("result.txt"); goto cleanup; }
    for (size_t i = 0; i < unique; ++i) fprintf(output, "%s\t%zu\n", sorted[i]->word, sorted[i]->count);
    int bad_write = ferror(output);
    if (fclose(output) || bad_write) { fprintf(stderr, "Could not write result.txt.\n"); goto cleanup; }
    printf("Words: %zu | Unique: %zu | Threads: %zu | Chunk bytes: %zu\n", total, unique, n, work.chunk);
    printf("Count, merge and sort: %.6f s\n", elapsed);
    for (size_t i = 0; i < n; ++i)
        printf("Thread %zu: %zu slices, %zu words\n", i, workers[i].slices, workers[i].words);
    printf("Alphabetical frequencies saved to result.txt\n");
    status = 0;
cleanup:
    if (workers) for (size_t i = 0; i < n; ++i) free_table(&workers[i].table);
    free_table(&merged); free(sorted); free(workers); free(threads); free(text);
    pthread_mutex_destroy(&work.lock);
    return status;
}
