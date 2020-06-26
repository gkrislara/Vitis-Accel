#include "imgproc/xf_custom_convolution.hpp"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#define V4L2_PIX_FMT_YUYV	0x56595559
#define WIDTH			1920
#define HEIGHT			1080
#define PPC			XF_NPPC1
extern "C" {

void read_f2d_input(XF_TNAME(XF_16UC1, PPC) *frm_in,
		    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &luma_out,
		    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &uv_out,
		    uint32_t fourcc)
{
	uint16_t lumamask    = (V4L2_PIX_FMT_YUYV == fourcc)? 0x00FF : 0xFF00;
	uint16_t lumashift   = (V4L2_PIX_FMT_YUYV == fourcc)? 0      : 8;
	uint16_t chromamask  = (V4L2_PIX_FMT_YUYV == fourcc)? 0xFF00 : 0x00FF;
	uint16_t chromashift = (V4L2_PIX_FMT_YUYV == fourcc)? 8      : 0;
	uint32_t steps = luma_out.rows * luma_out.cols / PPC;

read_loop:
	for (uint32_t i = 0; i < steps; i++) {
#pragma HLS pipeline II=1

		XF_TNAME(XF_16UC1, PPC) pixels = frm_in[i];
		XF_TNAME(XF_8UC1, PPC) ytmp  = 0;
		XF_TNAME(XF_8UC1, PPC) uvtmp = 0;

read_loop_inner:
		for(uint32_t j = 0; j < PPC; j++){

			XF_TNAME(XF_8UC1, PPC) ypix  = (pixels & lumamask) >> lumashift;
			XF_TNAME(XF_8UC1, PPC) uvpix = (pixels & chromamask) >> chromashift;

			ytmp  |=  ypix  << (8 * j);
			uvtmp |=  uvpix << (8 * j);

			pixels >>= 16;
		}

		luma_out.write(i, ytmp);
		uv_out.write(i, uvtmp);
	}
}

void write_f2d_output(xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &luma_in,
		      xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &uv_in,
		      XF_TNAME(XF_16UC1, PPC) *frm_out,
		      uint32_t fourcc)
{
	uint16_t lumashift   = (V4L2_PIX_FMT_YUYV == fourcc)? 0 : 8;
	uint16_t chromashift = (V4L2_PIX_FMT_YUYV == fourcc)? 8 : 0;
	uint32_t steps = luma_in.rows * luma_in.cols / PPC;

write_loop:
	for (uint32_t i = 0; i < steps; i++) {
#pragma HLS pipeline II=1

		XF_TNAME(XF_8UC1, PPC) ypixels  = luma_in.read(i);
		XF_TNAME(XF_8UC1, PPC) uvpixels = uv_in.read(i);
		XF_TNAME(XF_16UC1, PPC) pixels = 0;

write_loop_inner:
		for(uint32_t j = 0; j < PPC; j++){

			XF_TNAME(XF_16UC1, PPC) ytmp  = (ypixels  & 0xFF) << lumashift;
			XF_TNAME(XF_16UC1, PPC) uvtmp = (uvpixels & 0xFF) << chromashift;

			pixels |= ((ytmp | uvtmp) << (16 * j));

			ypixels  >>= 8;
			uvpixels >>= 8;
		}

		frm_out[i] = pixels;
	}
}

void copy_stream_dummy(xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &src,
		       xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> &dst)
{
	uint32_t steps = src.rows * src.cols / PPC;

copy_stream_loop:
	for (uint32_t i = 0; i < steps; i++) {
#pragma HLS pipeline II=1
		dst.write(i, src.read(i));
	}
}

void filter2d_accel(XF_TNAME(XF_16UC1, PPC) *frm_in, XF_TNAME(XF_16UC1, PPC) *frm_out, int16_t *kern, uint32_t rows, uint32_t cols, uint32_t fourcc_in,uint32_t fourcc_out)
{
#pragma HLS INTERFACE m_axi	port=frm_in	offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi	port=frm_out	offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi	port=kernel	offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=frm_in	bundle=control
#pragma HLS INTERFACE s_axilite port=frm_out	bundle=control
#pragma HLS INTERFACE s_axilite	port=kernel	bundle=control

#pragma HLS INTERFACE s_axilite port=rows	bundle=control
#pragma HLS INTERFACE s_axilite port=cols	bundle=control
#pragma HLS INTERFACE s_axilite port=fourcc_in	bundle=control
#pragma HLS INTERFACE s_axilite port=fourcc_out	bundle=control
#pragma HLS INTERFACE s_axilite port=return	bundle=control

	const uint32_t pCOLS = WIDTH / PPC;
	const uint32_t pCOLSD = WIDTH * 2 / PPC;

	xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> luma_in(rows, cols);
#pragma HLS stream variable=luma_in.data depth=pCOLS
	xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> uv_in(rows, cols);
#pragma HLS stream variable=uv_in.data depth=pCOLSD
	xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> uv_out(rows, cols);
#pragma HLS stream variable=uv_out.data depth=pCOLSD
	xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, PPC> luma_out(rows, cols);
#pragma HLS stream variable=luma_out.data depth=pCOLS

#pragma HLS DATAFLOW

	read_f2d_input((XF_TNAME(XF_16UC1, PPC)*) frm_in, luma_in, uv_in,
			fourcc_in);

	copy_stream_dummy(uv_in, uv_out);

	xf::cv::filter2D<XF_BORDER_CONSTANT, XF_FILTER_3X3, XF_FILTER_3X3, XF_8UC1,
		XF_8UC1, HEIGHT, WIDTH, PPC> (luma_in, luma_out, kern, 0);

	write_f2d_output(luma_out, uv_out, (XF_TNAME(XF_16UC1, PPC)*) frm_out,
			fourcc_out);
}
}
