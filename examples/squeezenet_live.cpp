#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <syslog.h>
#include <stdbool.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

#include "loader.h"

#include "squeezenet_lib.h"

#ifdef DEBUG
#define D(x)    x
#else
#define D(x)
#endif

#define LOGINFO(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOGERR(fmt, args...)     { syslog(LOG_CRIT, fmt, ## args); fprintf(stderr, fmt, ## args); }

static int quit = false;

/*
 * Assumes stride = width and subsampling = 2
 * Subsampels the whole image to half..
 */
void ycbcr2rgb(unsigned char *yy, unsigned char *cb, unsigned char *cr,
               unsigned char *rgb_out, int width, int height,
               int crop_width, int crop_height) {
  double r, g, b;
  int x, y;

  for (y = 0; y < crop_height; y ++) {
    for (x = 0; x < crop_width; x ++) {
        r = yy[2 * y * width + x * 2] + (1.4065 * (cr[y * width/2 + x] - 128));
        g = yy[2 * y * width + x * 2] - (0.3455 * (cb[y * width/2 + x] - 128)) - (0.7169 * (cr[y * width/2 + x] - 128));
        b = yy[2 * y * width + x * 2] + (1.7790 * (cb[y * width/2 + x] - 128));

        rgb_out[(y * crop_width + x) * 3 + 0] = r < 0 ? 0 : (r > 255 ? 255 : (unsigned char)r);
        rgb_out[(y * crop_width + x) * 3 + 1] = g < 0 ? 0 : (g > 255 ? 255 : (unsigned char)g);
        rgb_out[(y * crop_width + x) * 3 + 2] = b < 0 ? 0 : (b > 255 ? 255 : (unsigned char)b);
    }
  }
}

static void write_ppm(void *data, int width, int height, char *fname) {
  FILE *fp;
  int row, column, ch;
  fp = fopen(fname, "wb");
  if (!fp) {
    printf("Failed to open file: %s!!\n", fname);
    return;
  }
  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", width, height);
  fprintf(fp, "%d\n", 255);
  for (row = 0; row < height; row++)
    for (column = 0; column < width; column++)
      for (ch = 0; ch < 3; ch++)
        fputc (((unsigned char *) data)[(row * width + column) * 3 + ch], fp);
  fclose(fp);
}

void sig_cb(int num) {
  quit = true;
}

int main(int argc, char** argv) {
  struct timeval tv_start, tv_end;
  int            msec;

  ldr_image_t img;
  img.y  = NULL;
  img.cb = NULL;
  img.cr = NULL;

  int wout = 227;
  int hout = 227;

  int images = 0;

  signal(SIGKILL, sig_cb);
  signal(SIGINT, sig_cb);

  openlog("vidcap", LOG_PID | LOG_CONS, LOG_USER);

  // *********** INIT SQUEEZENET ************* //

  LOGINFO("Creating a SqueezeNet....\n");
  squeezenet_create();
  LOGINFO("OK!\n");

  // *********** INIT IMAGE STREAM *********** //

  ldr_init();

  fprintf(stderr, "Creating loader test\n");

  loader_t* ldr = ldr_create("/dev/ycbcr0", "640x480", 1); // 1 for color

  if (ldr == NULL) {
      fprintf(stderr, "create failed\n");
      goto EXIT;
  }

  // ********** RUN LIVE *************** //

  LOGINFO("Start grab images\n");

  gettimeofday(&tv_start, NULL);

  while (!quit) {

    if (!ldr_load(ldr, &img)) {
        LOGERR("load failed\n");
        goto EXIT;
    }

    unsigned char *rgb = (unsigned char *)malloc(wout * hout * 3);
    ycbcr2rgb(img.y, img.cb, img.cr, rgb, img.width, img.height, wout, hout);

    write_ppm(rgb, wout, hout, (char *)"rgb.ppm");

    squeezenet_classify(rgb);

    free(rgb);

    ldr_image_release(ldr, img.y);
    ldr_image_release(ldr, img.cb);
    ldr_image_release(ldr, img.cr);

    images ++;

  }

  // *********** TEAR DOWN ************* //

  gettimeofday(&tv_end, NULL);

  /* calculate fps */
  msec  = tv_end.tv_sec * 1000 + tv_end.tv_usec / 1000;
  msec -= tv_start.tv_sec * 1000 + tv_start.tv_usec / 1000;

  LOGINFO("Fetched %d images in %d milliseconds, fps:%0.3f\n",
          images,
          msec,
          (float)(images / (msec / 1000.0f)));

  closelog();

EXIT:
  ldr_delete(ldr);
  ldr_terminate();
  return EXIT_SUCCESS;
}
