#include "immintrin.h"

#include <stdint.h>
#include <X11/Xlib.h>
#include <time.h>
#include <stdio.h>

#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")

//=================================================================================================

const float ROI_X = -1.325f,
            ROI_Y = 0;

const float  dx    = 1/800.f, dy = 1/800.f;

const int32_t WIDTH  = 1920;
const int32_t HEIGHT = 1080;

//=================================================================================================

inline void FPSCounter(time_t* newTime, time_t* curTime, uint32_t* framesCounter) {
    if (((*newTime = time(NULL)) - *curTime) >= 1) {
        printf("FPS: %u\n", *framesCounter);

        *curTime = *newTime;
        *framesCounter = 0;
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

int main() {
    Display* d = XOpenDisplay(0);
    int screen = XDefaultScreen(d);
    int dplanes = DisplayPlanes(d, screen);
    Visual* defVis = XDefaultVisual(d, screen);
    XImage* image  = XCreateImage(d, defVis, dplanes, ZPixmap, 0, (char*)malloc(WIDTH * HEIGHT * sizeof(int32_t)), WIDTH, HEIGHT, 16, 0);

    Window w = XCreateWindow(d, DefaultRootWindow(d), 0, 0, WIDTH, HEIGHT, 0, CopyFromParent, CopyFromParent, 0, 0, CopyFromParent);
    XSelectInput(d, w, KeyPressMask | KeyReleaseMask);
    XMapWindow(d, w);
    XGCValues values;
    GC gc = XCreateGC(d, w, 0, &values);

    XEvent event;

    uint32_t framesCounter = 0;
    time_t curTime = time(NULL);
    time_t newTime = 0;

    const int    nMax  = 2048;
    const __m256 r2Max = _mm256_set1_ps (100.f);
    const __m256 _255  = _mm256_set1_ps (255.f);
    const __m256 _76543210 = _mm256_set_ps  (7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    
    const __m256 nmax  = _mm256_set1_ps (nMax);

    float xC = 0.f, yC = 0.f, scale = 1.f;

    uint8_t global_program_flag = 1;
    while (global_program_flag) {
        XCheckMaskEvent(d, KeyPressMask, &event);
        ProcessEvent(&event, &xC, &yC, &scale);


        #pragma omp parallel for
        for (int iy = 0; iy < HEIGHT * global_program_flag; iy++) {
            float y0 = ( ((float)iy - 300.f) * dy ) * scale + ROI_Y + yC;

            #pragma omp parallel for
            for (int ix = 0; ix < WIDTH; ix += 8/*, x0 += dx * 8 * scale*/) { 
                float x0 = ( (     (float)  ix   - 400.f) * dx ) * scale + ROI_X + xC;
                __m256 X0 = _mm256_add_ps (_mm256_set1_ps (x0), _mm256_mul_ps (_76543210, _mm256_set1_ps (dx * scale)));
                __m256 Y0 =             _mm256_set1_ps (y0);

                __m256 X = X0, Y = Y0;
                
                __m256i N = _mm256_setzero_si256();
                
                uint8_t local_program_flag = 1;
                for (int n = 0; n < nMax; n++) {
                    if (local_program_flag) {
                        __m256 x2 = _mm256_mul_ps (X, X),
                           y2 = _mm256_mul_ps (Y, Y);
                           
                        __m256 r2 = _mm256_add_ps (x2, y2);

                        __m256 cmp = _mm256_cmp_ps (r2, r2Max, _CMP_LE_OQ);
                        int mask   = _mm256_movemask_ps (cmp);
                        if (!mask) local_program_flag = 0;

                        N = _mm256_sub_epi32 (N, _mm256_castps_si256 (cmp));

                        __m256 xy = _mm256_mul_ps (X, Y);

                        X = _mm256_add_ps (_mm256_sub_ps (x2, y2), X0);
                        Y = _mm256_add_ps (_mm256_add_ps (xy, xy), Y0);
                    }
                }

                __m256 I = _mm256_mul_ps (_mm256_sqrt_ps (_mm256_sqrt_ps (_mm256_div_ps (_mm256_cvtepi32_ps (N), nmax))), _255);

                for (int i = 0; i < 8; i++) {
                    int*   pn = (int*)   &N;
                    float* pI = (float*) &I;

                    uint8_t color = (uint8_t)pI[i];

                    *((int*)image->data + iy * WIDTH + ix + i) = 0xff000000 + (color << 16) + (((color & 1) * 0x40 + (1 - (color & 1)) * 0xb0) << 8) + (265 - color);
                }
            }
        }
            
        framesCounter++;
        FPSCounter(&newTime, &curTime, &framesCounter);

        XPutImage(d, w, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(d);
    }
}
