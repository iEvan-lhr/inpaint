#ifndef TEXTINPAINT_BRIDGE_H
#define TEXTINPAINT_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif
int tp_prepare(const unsigned char*, int, int, int, int, int, int, int, double, int, int, float*, char*, int);
int tp_boxes(const float*, int, int, int, int, float, float, float, int, float*, int*, char*, int);
int tp_text_heights(const unsigned char*, int, int, int, const float*, int, float*, char*, int);
int tp_graphic_panels(const unsigned char*, int, int, int, int*, int, int*, char*, int);
int tp_mask(const unsigned char*, int, int, int, const float*, int, float, int, int, int, unsigned char*, char*, int);
int tp_inpaint(const unsigned char*, int, int, int, const unsigned char*, double, unsigned char*, char*, int);
int tp_overlay(const unsigned char*, int, int, int, const float*, int, unsigned char*, char*, int);
#ifdef __cplusplus
}
#endif
#endif
