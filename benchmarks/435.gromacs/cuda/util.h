#ifndef GROMACS_UTIL
#define GROMACS_UTIL

typedef unsigned int uint32_t;
typedef int           int32_t;


bool readOpenBracket(FILE* fp) {
    while ( fgetc(fp) != '[' ) ;
    return true;
}
bool readCloseBracket(FILE* fp) {
    while ( fgetc(fp) != ']' ) ;
    return true;
}
bool readCommaOrCloseBracket(FILE* fp, char* res) {
    char c = fgetc(fp);
    while(c != ',' && c != ']')
        c = fgetc(fp);
    *res = c;
    return true;
}
bool readI32(FILE* fp, int* res) {
    // in case of failure, it does not advances the cursor
    return (0 != fscanf(fp, "%d", res));
}

unsigned int readArray1dI32 (FILE* fp, const unsigned int maxlen, int* res) {
    int  data = 0;
    char chr = '0';
    bool goOn = true;
    unsigned int count = 0;
    readOpenBracket(fp);
   
    do {
        bool success1 = readI32(fp, &data);
        bool success2 = readCommaOrCloseBracket(fp, &chr);

        if(success1) { res[count++] = data; }
        if(success2 && (chr == ']')) { goOn = false; }

        if ( (!success1) && success2 && (chr == ',')) {
            fprintf(stderr, "Malformat input in readArray1dI32, Exiting!!!\n");
            exit(1);
        }
    } while (goOn && (count < maxlen));

    return count;
}

bool readDataset( char* file_name
                , int* arr1, const unsigned int maxlen1, int* len1
                , int* arr2, const unsigned int maxlen2, int* len2
                , int* arr3, const unsigned int maxlen3, int* len3
                , int* arr4, const unsigned int maxlen4, int* len4
                , int* arr5, const unsigned int maxlen5, int* len5
                ) {
    FILE* fp = fopen(file_name, "r");

    *len1 = readArray1dI32 (fp, maxlen1, arr1);
    *len2 = readArray1dI32 (fp, maxlen2, arr2);
    *len3 = readArray1dI32 (fp, maxlen3, arr3);
    *len4 = readArray1dI32 (fp, maxlen4, arr4);
    *len5 = readArray1dI32 (fp, maxlen5, arr5);
#if 0
    fprintf(stderr, "\nArray with %d elements, who are: \n", len1);
    for(int i=0; i<len1; i++) {
        fprintf(stderr, "%d, ", arr1[i]);
    }
    fprintf(stderr, "\n");
#endif

    fclose (fp);
    return true;
}


/***********************/
/*** Various Helpers ***/
/***********************/
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

void randomInit(real* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = ((real) rand()) / ((real)RAND_MAX);
}

void printArray(real* data, int size) {
    printf("[");
    if(size > 0) printf("%f", data[0]);
    for (int i = 1; i < size; ++i)
        printf(", %f", data[i]);
    printf("]");
}

template<class T>
void zeroOut(T* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = 0;
}

#define EPS 0.0003
bool validate32(real* A, real* B, unsigned int sizeAB) {
    for(unsigned int i = 0; i < sizeAB; i++) {
        real error = fabs(A[i] - B[i]) / fabs(A[i]);
        if (error > EPS) {
            printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

bool validate64(double* A, double* B, unsigned int sizeAB) {
    for(unsigned int i = 0; i < sizeAB; i++) {
        if (fabs(A[i] - B[i]) > EPS) {
            printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

#endif // define GROMACS_UTIL
