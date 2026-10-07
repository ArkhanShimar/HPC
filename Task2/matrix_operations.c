#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

typedef struct { int rows, cols; double *data; } Matrix;
static size_t line_number;

// Read a physical line so a missing value cannot be taken from the next row.
static int read_line(FILE *fp, char **out) {
    size_t used = 0, capacity = 128;
    char *line = malloc(capacity), c;
    if (!line) return -1;
    int rc;
    while ((rc = fscanf(fp, "%c", &c)) == 1 && c != '\n') {
        if (used + 1 == capacity) {
            if (capacity > SIZE_MAX / 2) { free(line); return -1; }
            char *larger = realloc(line, capacity * 2);
            if (!larger) { free(line); return -1; }
            line = larger; capacity *= 2;
        }
        line[used++] = c;
    }
    if (ferror(fp)) { free(line); return -1; }
    if (rc != 1 && !used) { free(line); return 0; }
    line[used] = '\0'; *out = line; line_number++; return 1;
}
static char *skip_space(char *p) { while (isspace((unsigned char)*p)) p++; return p; }
static int allocate(Matrix *m, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || (size_t)rows > SIZE_MAX / (size_t)cols / sizeof(double)) return 0;
    m->rows = rows; m->cols = cols;
    m->data = malloc((size_t)rows * cols * sizeof(double));
    return m->data != NULL;
}
static int read_matrix(FILE *fp, Matrix *m) {
    char *line = NULL, *p, *end; int rc;
    do {
        free(line); line = NULL; rc = read_line(fp, &line);
        if (rc <= 0) return rc;
        p = skip_space(line);
    } while (!*p);
    errno = 0; long rows = strtol(p, &end, 10);
    if (p == end || errno || rows <= 0 || rows > INT_MAX) goto invalid;
    p = skip_space(end); if (*p++ != ',') goto invalid;
    p = skip_space(p); errno = 0; long cols = strtol(p, &end, 10);
    if (p == end || errno || cols <= 0 || cols > INT_MAX || *skip_space(end)) goto invalid;
    free(line); line = NULL;
    if (!allocate(m, (int)rows, (int)cols)) {
        fprintf(stderr, "Line %zu: matrix too large or allocation failed.\n", line_number); return -1;
    }
    for (int r = 0; r < m->rows; ++r) {
        rc = read_line(fp, &line);
        if (rc != 1) { fprintf(stderr, "Missing row %d after line %zu.\n", r+1, line_number); return -1; }
        p = line;
        for (int c = 0; c < m->cols; ++c) {
            p = skip_space(p); errno = 0;
            double value = strtod(p, &end);
            if (p == end || errno || !isfinite(value)) goto invalid;
            m->data[(size_t)r*m->cols+c] = value;
            p = skip_space(end);
            if (c + 1 < m->cols) { if (*p != ',') goto invalid; p++; }
            else if (*p) goto invalid;
        }
        free(line); line = NULL;
    }
    return 1;
invalid:
    fprintf(stderr, "Line %zu: invalid header, numeric value or row length.\n", line_number);
    free(line); return -1;
}
static void write_matrix(FILE *out, const char *name, const Matrix *m) {
    fprintf(out, "%s - %d,%d\n", name, m->rows, m->cols);
    for (int r = 0; r < m->rows; ++r) {
        for (int c = 0; c < m->cols; ++c) {
            double v = m->data[(size_t)r*m->cols+c];
            if (isnan(v)) fprintf(out, "NaN"); else fprintf(out, "%.12g", v);
            fprintf(out, "%s", c+1 == m->cols ? "\n" : ",");
        }
    }
    fprintf(out, "\n");
}
static int operation(FILE *out, const Matrix *a, const Matrix *b, int op, int requested) {
    const char *names[] = {"Addition", "Subtraction", "Element-wise multiplication", "Element-wise division", "Transpose A", "Transpose B", "Matrix multiplication"};
    if (op < 4 && (a->rows != b->rows || a->cols != b->cols)) {
        fprintf(out, "%s cannot be done (shapes differ).\n", names[op]); return 1;
    }
    if (op == 6 && a->cols != b->rows) {
        fprintf(out, "%s cannot be done (A.cols != B.rows).\n", names[op]); return 1;
    }
    const Matrix *source = op == 5 ? b : a;
    int rows = op == 4 || op == 5 ? source->cols : a->rows;
    int cols = op == 4 || op == 5 ? source->rows : (op == 6 ? b->cols : a->cols);
    Matrix result = {0};
    if (!allocate(&result, rows, cols)) { fprintf(stderr, "Result allocation failed.\n"); return 0; }
    int threads = requested < rows ? requested : rows;
    // Each iteration owns one complete output row; no shared accumulator is needed.
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            size_t i = (size_t)r*cols+c;
            if (op == 0) result.data[i] = a->data[i] + b->data[i];
            else if (op == 1) result.data[i] = a->data[i] - b->data[i];
            else if (op == 2) result.data[i] = a->data[i] * b->data[i];
            else if (op == 3) result.data[i] = b->data[i] == 0 ? NAN : a->data[i] / b->data[i];
            else if (op == 4 || op == 5) result.data[i] = source->data[(size_t)c*source->cols+r];
            else {
                double sum = 0;
                for (int k = 0; k < a->cols; ++k) sum += a->data[(size_t)r*a->cols+k] * b->data[(size_t)k*b->cols+c];
                result.data[i] = sum;
            }
        }
    }
    printf("  %s: %dx%d, thread limit %d\n", names[op], rows, cols, threads);
    write_matrix(out, names[op], &result); free(result.data);
    return !ferror(out);
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s input.txt threads\n", argv[0]); return 1; }
    char *end; errno = 0; long threads = strtol(argv[2], &end, 10);
    if (errno || end == argv[2] || *end || threads < 1 || threads > INT_MAX) {
        fprintf(stderr, "Thread count must be a positive integer.\n"); return 1;
    }
    FILE *input = fopen(argv[1], "r");
    if (!input) { perror(argv[1]); return 1; }
    FILE *out = fopen("results.txt", "w");
    if (!out) { perror("results.txt"); fclose(input); return 1; }
    Matrix a = {0}, b = {0}; int status = 1, pairs = 0, rc;
    omp_set_dynamic(0);
    while ((rc = read_matrix(input, &a)) == 1) {
        if (read_matrix(input, &b) != 1) { fprintf(stderr, "Incomplete or invalid matrix pair %d.\n", pairs+1); goto cleanup; }
        pairs++;
        fprintf(out, "Pair %d | A: %d,%d | B: %d,%d\n\n", pairs, a.rows, a.cols, b.rows, b.cols);
        printf("Pair %d: A %dx%d, B %dx%d\n", pairs, a.rows, a.cols, b.rows, b.cols);
        for (int op = 0; op < 7; ++op) if (!operation(out, &a, &b, op, (int)threads)) goto cleanup;
        free(a.data); free(b.data); a.data = b.data = NULL;
    }
    if (rc < 0 || !pairs) { fprintf(stderr, "Input does not contain complete valid matrix pairs.\n"); goto cleanup; }
    if (ferror(out)) goto cleanup;
    status = 0;
cleanup:
    free(a.data); free(b.data);
    if (fclose(input)) status = 1;
    if (fclose(out)) status = 1;
    if (!status) printf("Processed %d pairs. Saved results.txt\n", pairs);
    if (status) remove("results.txt");
    return status;
}
