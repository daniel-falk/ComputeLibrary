#ifndef __NEON_SQUEEZENET__H
#define __NEON_SQUEEZENET__H

void squeezenet_create();
void squeezenet_set_rgb(unsigned char *img_buffer);
void squeezenet_set_ybcbr_planar(unsigned char *yy, unsigned char *cb, unsigned char *cr, int stride);
void squeezenet_classify();

#endif /* __NEON_SQUEEZENET__H */

