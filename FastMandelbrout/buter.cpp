#include "X11/Xlib.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#include <immintrin.h>

const int32_t WIDTH  = 1920;
const int32_t HEIGHT = 1072;

const int32_t amountMax = 256;
const float dx = 1 / (float)WIDTH;
const float dy = 1 / (float)WIDTH;
const float MAX_R_SQUARED = 100;

const float START_X = -1.325;
const float START_Y = 0;

const int32_t THREAD_NUM = 16;

struct burgerArgs {
    float xShift;
    float scale;
    float yShift;
    char* data;
    float y0;

    uint32_t curY;
};

void ProcessEvent(XEvent* event, float* xShift, float* yShift, float* scale);
void* ButerCalculate(void* args);
void FPSCounter(time_t* newTime, time_t* curTime, uint32_t* framesCounter);

int main(void) {
    Display* d = XOpenDisplay(0);
    int screen = XDefaultScreen(d);
    int dplanes = DisplayPlanes(d, screen);
    Visual* defVis = XDefaultVisual(d, screen);
    XImage* image  = XCreateImage(d, defVis, dplanes, ZPixmap, 0, (char*)malloc(WIDTH * HEIGHT * sizeof(int32_t)), WIDTH, HEIGHT, 16, 0);

    if (d) {
        Window w = XCreateWindow(d, DefaultRootWindow(d), 0, 0, WIDTH, HEIGHT, 0, CopyFromParent, CopyFromParent, 0, 0, CopyFromParent);
        XSelectInput(d, w, KeyPressMask | KeyReleaseMask);
        XMapWindow(d, w);
        XGCValues values;
        GC gc = XCreateGC(d, w, 0, &values);

        float xShift = 0, yShift = 0;
        float scale = 1.f;
    
        XEvent event;

        uint32_t framesCounter = 0;
        time_t curTime = time(NULL);
        time_t newTime = 0;

        burgerArgs arg[THREAD_NUM] = {};
        for (uint32_t curThread = 0; curThread < THREAD_NUM; curThread++) {
            arg[curThread].data   = image->data;
        }

        pthread_t* threads = (pthread_t*)calloc(THREAD_NUM, sizeof(pthread_t));

        for(;;) {  
            XCheckMaskEvent(d, KeyPressMask, &event);
            ProcessEvent(&event, &xShift, &yShift, &scale);

            for (uint32_t curThread = 0; curThread < THREAD_NUM; curThread++) {
                arg[curThread].scale  = scale;
                arg[curThread].xShift = xShift;
                arg[curThread].yShift = yShift;
            }

            for (uint32_t curString = 0; curString < HEIGHT; curString += THREAD_NUM) {
                for (uint32_t curThread = 0; curThread < THREAD_NUM; curThread++) {
                    arg[curThread].y0 = ((float)curString + (float)curThread - (float)HEIGHT / 2) * dy * scale + yShift + START_Y;
                    arg[curThread].curY = curString + curThread;

                    pthread_create(threads + curThread, NULL, ButerCalculate, (void*)(arg + curThread));
                }

                for (uint32_t curThread = 0; curThread < THREAD_NUM; curThread ++) {
                    pthread_join(threads[curThread], NULL);
                }
            }

            framesCounter++;
            FPSCounter(&newTime, &curTime, &framesCounter);

            XPutImage(d, w, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
            XFlush(d);
        }

        free(threads);
    }
}

inline void ProcessEvent(XEvent* event, float* xShift, float* yShift, float* scale) {
    if (event->type == KeyPress) {
        switch (event->xkey.keycode) {
            case 0x72:
                *xShift += dx * 10.f;
                break;
            case 0x71:
                *xShift -= dx * 10.f;
                break;
            case 0x6f:
                *yShift -= dy * 10.f;
                break;
            case 0x74:
                *yShift += dy * 10.f;
                break;
            case 0x34: 
                *scale += dx * 10.f;
                break;
            case 0x35:
                *scale -= dx * 10.f;
                break;
        }

        event->type = KeyRelease;
    }
}

void* ButerCalculate(void* args) {
    static const __m256 _76543210 = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f),
                        maxRSq    = _mm256_set1_ps(MAX_R_SQUARED),
                        _255      = _mm256_set1_ps(255.f),
                        maxN      = _mm256_set1_ps(amountMax);

    burgerArgs* ptr = (burgerArgs*)args;     

    float x0 = ( - (float)WIDTH / 2) * dx * ptr->scale + ptr->xShift + START_X;
    for (uint32_t curX = 0; curX < WIDTH; curX += 8, x0 += 8 * dx * ptr->scale) {
        __m256 X0 = _mm256_add_ps(_mm256_set1_ps(x0), _mm256_mul_ps(_76543210, _mm256_set1_ps(dx * ptr->scale))),
               Y0 = _mm256_set1_ps(ptr->y0);

        __m256 X = X0, Y = Y0;

        __m256i N = _mm256_setzero_si256();

        int32_t flag = 1;
        for (uint32_t curIter = 0; curIter < (amountMax * flag); curIter++) {
            __m256  xSq = _mm256_mul_ps(X, X),
                    ySq = _mm256_mul_ps(Y, Y),
                    RSq = _mm256_add_ps(xSq, ySq);

            __m256 cmp = _mm256_cmp_ps(RSq, maxRSq, _CMP_LE_OQ);
            int32_t mask = _mm256_movemask_ps(cmp);

            if (mask == 0)
                flag = 0;

            N   = _mm256_sub_epi32(N, _mm256_castps_si256(cmp));
            __m256 xy = _mm256_mul_ps(X, Y);

            X = _mm256_add_ps(_mm256_sub_ps(xSq, ySq), X0);
            Y = _mm256_add_ps(_mm256_add_ps(xy, xy), Y0);
        }

        __m256 I = _mm256_mul_ps(_mm256_sqrt_ps(_mm256_sqrt_ps(_mm256_div_ps(_mm256_cvtepi32_ps(N), maxN))),_255);

        for (uint32_t iter = 0; iter < 8; iter++) {
            float*  pI         = (float*) &I;

            uint8_t color = (uint8_t)pI[iter];

            *((int*)ptr->data + ptr->curY * WIDTH + curX + iter) = 0xff000000 + (color << 16) + (((color & 1) * 0x40 + (1 - (color & 1)) * 0xb0) << 8) + (265 - color);
        }
    }

    pthread_exit(NULL);
}

inline void FPSCounter(time_t* newTime, time_t* curTime, uint32_t* framesCounter) {
    if (((*newTime = time(NULL)) - *curTime) >= 1) {
        printf("FPS: %u\n", *framesCounter);

        *curTime = *newTime;
        *framesCounter = 0;
    }
}
