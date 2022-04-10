
// g+++ 7-SSEimgAlpha.cpp -O3 -msse4.2 
#include "X11/Xlib.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>

#include <immintrin.h>
#include <emmintrin.h>

const char I = 255u,
           Z = 0x80u;
           
const int32_t tableSize  = 1920138;
const int32_t catSize    = 1920138;

const int32_t tableHeaderSize = 138;
const int32_t catHeaderSize   = 138;

const int32_t catWidth   = 800;
const int32_t catHeight  = 600;

const int32_t tableWidth  = 800;
const int32_t tableHeight = 600;

const int32_t outputWidth  = 800;
const int32_t outputHeight = 600;

const int32_t allighShift = 6;

//-------------------------------------------------------------------------------------------------

char* ReadFile(char* fileName, int32_t fileSize) {
    assert(fileName != NULL);

    int32_t table = open(fileName, O_RDONLY);
    char* tableBuffer = malloc(fileSize + 6);
    tableBuffer += allighShift;
    read(table, tableBuffer, fileSize);
    close(table);

    return tableBuffer;
}

int main() {
    const __m128i   _0 =                   _mm_set_epi8(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    const __m128i _255 = _mm_cvtepu8_epi16(_mm_set_epi8(I,I,I,I, I,I,I,I, I,I,I,I, I,I,I,I));

    char* tableBuffer = ReadFile("Table.bmp", tableSize);
    char* catBuffer   = ReadFile("Cat.bmp", catSize);
    
    tableBuffer  += tableHeaderSize;
    catBuffer    += catHeaderSize;

    Display* d = XOpenDisplay(0);
    int screen = XDefaultScreen(d);
    int dplanes = DisplayPlanes(d, screen);
    Visual* defVis = XDefaultVisual(d, screen);
    XImage* image  = XCreateImage(d, defVis, dplanes, ZPixmap, 0, malloc(outputWidth * outputHeight * sizeof(int32_t)), outputWidth, outputHeight, 16, 0);

    Window w = XCreateWindow(d, DefaultRootWindow(d), 0, 0, outputWidth, outputHeight, 0, CopyFromParent, CopyFromParent, 0, 0, CopyFromParent);
    XMapWindow(d, w);

    XGCValues values;
    GC gc = XCreateGC(d, w, 0, &values);

    uint32_t framesCounter = 0;
    time_t curTime = time(NULL);
    time_t newTime = 0;

    for (;;) {

        for (int y = 0; y < tableHeight; y++) {
            for (int x = 0; x < tableWidth; x += 4) {
                __m128i fr = _mm_load_si128((__m128i*)((int32_t*)catBuffer + y * tableWidth + x));                   // fr = front[y][x]
                __m128i bk = _mm_load_si128((__m128i*)((int32_t*)tableBuffer + y * tableWidth + x));

                __m128i FR = (__m128i) _mm_movehl_ps ((__m128) _0, (__m128) fr);       // FR = (fr >> 8*8)
                __m128i BK = (__m128i) _mm_movehl_ps ((__m128) _0, (__m128) bk);

                fr = _mm_cvtepu8_epi16(fr);                  // fr[i] = (WORD) fr[i]
                FR = _mm_cvtepu8_epi16(FR);

                bk = _mm_cvtepu8_epi16(bk);
                BK = _mm_cvtepu8_epi16(BK);

                __m128i moveA = _mm_set_epi8(Z, 0xe, Z, 0xe, Z, 0xe, Z, 0xe, Z, 0x6, Z, 0x6, Z, 0x6, Z, 0x6);

                __m128i a = _mm_shuffle_epi8 (fr, moveA);                                // a [for r0/b0/b0...] = a0...
                __m128i A = _mm_shuffle_epi8 (FR, moveA);
                
                fr = _mm_mullo_epi16(fr, a);                                           // fr *= a
                FR = _mm_mullo_epi16(FR, A);

                bk = _mm_mullo_epi16(bk, _mm_sub_epi16(_255, a));                                  // bk *= (255-a)
                BK = _mm_mullo_epi16(BK, _mm_sub_epi16(_255, A));

                __m128i sum = _mm_add_epi16(fr, bk);                                       // sum = fr*a + bk*(255-a)
                __m128i SUM = _mm_add_epi16(FR, BK);

                __m128i moveSum = _mm_set_epi8 (Z, Z, Z, Z, Z, Z, Z, Z, 0xf, 0xd, 0xb, 0x9, 0x7, 0x5, 0x3, 0x1);

                sum = _mm_shuffle_epi8(sum, moveSum);                                      // sum[i] = (sium[i] >> 8) = (sum[i] / 256)
                SUM = _mm_shuffle_epi8(SUM, moveSum);

                __m128i color = (__m128i) _mm_movelh_ps((__m128) sum, (__m128) SUM);  // color = (sumHi << 8*8) | sum
                _mm_store_si128((__m128i*)((int32_t*)image->data + y * tableWidth + x), color);
            }
        }
        framesCounter++;

        if (((newTime = time(NULL)) - curTime) >= 1) {
            printf("FPS: %u\n", framesCounter);

            curTime = newTime;
            framesCounter = 0;
        }

        XPutImage(d, w, gc, image, 0, 0, 0, 0, outputWidth, outputHeight);
        XFlush(d);
    }
}

    