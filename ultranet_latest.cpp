#define DEBUG

#ifdef DEBUG

#endif
#include <stdint.h>
#include <ap_int.h>
#include <fstream>
#include <iostream>
using namespace std;
#include "config_opt3.h"
#include "weights_opt3.hpp"

#include "conv1x1DSP2.hpp"
#include "conv2d_DSPopt3.hpp"
#include "conv2d_l0_opt.hpp"

#include "function.h"
#include "pool_reord.hpp"
#include "stream_tools.h"
#include "debug.hpp"





void do_compute2(ap_uint<8> *in, ap_uint<32> *out, const unsigned int reps = 1)
{
#pragma HLS DATAFLOW

const unsigned int num_per_rep = 160 * 320 * 3;

hls::stream<ap_uint<8>> in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable = in_stream_extract depth = 1024
ExtractPixels<8, num_per_rep>(in, in_stream_extract, reps);

hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH>> in_stream1("in_stream1");
#pragma HLS STREAM variable = in_stream1 depth = 16
StreamingDataWidthConverter_Batch<8, CONV_0_IN_BIT * CONV_0_IFM_CH, num_per_rep>(in_stream_extract, in_stream1, reps);
#ifdef DEBUG
cout << "in_stream1 size " << in_stream1.size() << endl;
#endif
//***********CONV_0*************
hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2>> conv_0_out("conv_0_out");
#pragma HLS STREAM variable = conv_0_out depth = 128
conv3x3_l0_bn_act_DSPopt<CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_IFM_CH, CONV_0_IN_BIT, CONV_0_OFM_CH, CONV_0_OUT_BIT, CONV_0_W_BIT, 98, CONV_0_INC_BIT_NEW, CONV_0_BIAS_BIT_NEW, CONV_0_SIMD_DSP6, 4, CONV_0_INPE, CONV_0_PE_DSP6, CONV_0_L_SHIFT>(in_stream1, conv_0_w_new, conv_0_inc_new, conv_0_bias_new, conv_0_out, reps);
//***********POOL_0*************
hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2>> pool_0_out("pool_0_out");
#pragma HLS STREAM variable = pool_0_out depth = 128
max_pool2x2<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH, CONV_0_OUT_BIT, CONV_0_PE_DSP6>(conv_0_out, pool_0_out, reps);
//***********CONV_1*************
hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_PE_DSP6 * 2>> conv_1_out("conv_1_out");
#pragma HLS STREAM variable = conv_1_out depth = 128
conv3x3_bn_act_DSPopt<CONV_1_IFM_ROW, CONV_1_IFM_COL, CONV_1_IFM_CH, CONV_1_IN_BIT, CONV_1_OFM_CH, CONV_1_OUT_BIT, CONV_1_W_BIT, 81, CONV_1_INC_BIT_NEW, CONV_1_BIAS_BIT_NEW, CONV_1_SIMD_DSP6, 4, CONV_1_INPE, CONV_1_PE_DSP6, CONV_1_L_SHIFT>(pool_0_out, conv_1_w_new, conv_1_inc_new, conv_1_bias_new, conv_1_out, reps);
//***********POOL_1*************
hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_PE_DSP6 * 2>> pool_1_out("pool_1_out");
#pragma HLS STREAM variable = pool_1_out depth = 128
max_pool2x2<CONV_1_OFM_ROW, CONV_1_OFM_COL, CONV_1_OFM_CH, CONV_1_OUT_BIT, CONV_1_PE_DSP6>(conv_1_out, pool_1_out, reps);
//***********CONV_2*************
hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE_DSP6 * 2>> conv_2_out("conv_2_out");
#pragma HLS STREAM variable = conv_2_out depth = 128
conv3x3_bn_act_DSPopt<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH, CONV_2_IN_BIT, CONV_2_OFM_CH, CONV_2_OUT_BIT, CONV_2_W_BIT, 32, CONV_2_INC_BIT_NEW, CONV_2_BIAS_BIT_NEW, CONV_2_SIMD_DSP6, 4, CONV_2_INPE, CONV_2_PE_DSP6, CONV_2_L_SHIFT>(pool_1_out, conv_2_w_new, conv_2_inc_new, conv_2_bias_new, conv_2_out, reps);
//***********POOL_2*************
hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE_DSP6 * 2>> pool_2_out("pool_2_out");
#pragma HLS STREAM variable = pool_2_out depth = 128
max_pool2x2<CONV_2_OFM_ROW, CONV_2_OFM_COL, CONV_2_OFM_CH, CONV_2_OUT_BIT, CONV_2_PE_DSP6>(conv_2_out, pool_2_out, reps);
//***********CONV_3*************
hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_PE_DSP6 * 2>> conv_3_out("conv_3_out");
#pragma HLS STREAM variable = conv_3_out depth = 128
conv3x3_bn_act_DSPopt<CONV_3_IFM_ROW, CONV_3_IFM_COL, CONV_3_IFM_CH, CONV_3_IN_BIT, CONV_3_OFM_CH, CONV_3_OUT_BIT, CONV_3_W_BIT, 36, CONV_3_INC_BIT_NEW, CONV_3_BIAS_BIT_NEW, CONV_3_SIMD_DSP6, 4, CONV_3_INPE, CONV_3_PE_DSP6, CONV_3_L_SHIFT>(pool_2_out, conv_3_w_new, conv_3_inc_new, conv_3_bias_new, conv_3_out, reps);
//***********POOL_3*************
hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_PE_DSP6 * 2>> pool_3_out("pool_3_out");
#pragma HLS STREAM variable = pool_3_out depth = 128
max_pool2x2<CONV_3_OFM_ROW, CONV_3_OFM_COL, CONV_3_OFM_CH, CONV_3_OUT_BIT, CONV_3_PE_DSP6>(conv_3_out, pool_3_out, reps);
//***********CONV_4*************
hls::stream<ap_uint<CONV_4_OUT_BIT * CONV_4_PE_DSP6 * 2>> conv_4_out("conv_4_out");
#pragma HLS STREAM variable = conv_4_out depth = 128
conv3x3_bn_act_DSPopt<CONV_4_IFM_ROW, CONV_4_IFM_COL, CONV_4_IFM_CH, CONV_4_IN_BIT, CONV_4_OFM_CH, CONV_4_OUT_BIT, CONV_4_W_BIT, 36, CONV_4_INC_BIT_NEW, CONV_4_BIAS_BIT_NEW, CONV_4_SIMD_DSP6, 4, CONV_4_INPE, CONV_4_PE_DSP6, CONV_4_L_SHIFT>(pool_3_out, conv_4_w_new, conv_4_inc_new, conv_4_bias_new, conv_4_out, reps);
//***********CONV_5*************
hls::stream<ap_uint<CONV_5_OUT_BIT * CONV_5_PE_DSP6 * 2>> conv_5_out("conv_5_out");
#pragma HLS STREAM variable = conv_5_out depth = 128
conv3x3_bn_act_DSPopt<CONV_5_IFM_ROW, CONV_5_IFM_COL, CONV_5_IFM_CH, CONV_5_IN_BIT, CONV_5_OFM_CH, CONV_5_OUT_BIT, CONV_5_W_BIT, 36, CONV_5_INC_BIT_NEW, CONV_5_BIAS_BIT_NEW, CONV_5_SIMD_DSP6, 4, CONV_5_INPE, CONV_5_PE_DSP6, CONV_5_L_SHIFT>(conv_4_out, conv_5_w_new, conv_5_inc_new, conv_5_bias_new, conv_5_out, reps);
//***********CONV_6*************
hls::stream<ap_uint<CONV_6_OUT_BIT * CONV_6_PE_DSP6 * 2>> conv_6_out("conv_6_out");
#pragma HLS STREAM variable = conv_6_out depth = 128
conv3x3_bn_act_DSPopt<CONV_6_IFM_ROW, CONV_6_IFM_COL, CONV_6_IFM_CH, CONV_6_IN_BIT, CONV_6_OFM_CH, CONV_6_OUT_BIT, CONV_6_W_BIT, 24, CONV_6_INC_BIT_NEW, CONV_6_BIAS_BIT_NEW, CONV_6_SIMD_DSP6, 4, CONV_6_INPE, CONV_6_PE_DSP6, CONV_6_L_SHIFT>(conv_5_out, conv_6_w_new, conv_6_inc_new, conv_6_bias_new, conv_6_out, reps);
//***********CONV_7*************
hls::stream<ap_uint<CONV_7_OUT_BIT * CONV_7_PE_DSP6 * 2>> conv_7_out("conv_7_out");
#pragma HLS STREAM variable = conv_7_out depth = 128
conv3x3_bn_act_DSPopt<CONV_7_IFM_ROW, CONV_7_IFM_COL, CONV_7_IFM_CH, CONV_7_IN_BIT, CONV_7_OFM_CH, CONV_7_OUT_BIT, CONV_7_W_BIT, 36, CONV_7_INC_BIT_NEW, CONV_7_BIAS_BIT_NEW, CONV_7_SIMD_DSP6, 4, CONV_7_INPE, CONV_7_PE_DSP6, CONV_7_L_SHIFT>(conv_6_out, conv_7_w_new, conv_7_inc_new, conv_7_bias_new, conv_7_out, reps);
//***********CONV_8*************
hls::stream<ap_uint<32 * CONV_8_PE_DSP2>> conv_8_out("conv_8_out");
#pragma HLS STREAM variable = conv_8_out depth = 64
conv1x1_DSPopt<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH, CONV_8_IN_BIT, CONV_8_OFM_CH, CONV_8_W_BIT, CONV_8_BIAS_BIT_NEW, 24, CONV_8_SIMD_DSP2, CONV_8_PE_DSP2, CONV_8_INPE>(conv_8_out, conv_8_w_new, conv_8_bias_new, conv_8_out, reps);
hls::stream<ap_uint<32>> out_stream("out_stream");
#pragma HLS STREAM variable = out_stream depth = 128
StreamingDataWidthConverter_Batch<32 * CONV_8_PE_DSP2, 32, CONV_8_OFM_ROW * CONV_8_OFM_COL * CONV_8_OFM_CH / 2>(conv_8_out, out_stream, reps);
AddLast<CONV_8_OFM_ROW * CONV_8_OFM_COL * CONV_8_OFM_CH>(out_stream, out, reps);
}
}
void ultra_net(ap_uint<8> *in, ap_uint<32> *out, const unsigned reps) {

#pragma HLS INTERFACE m_axi port=in bundle=gmem0 offset=slave depth=1024
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=128
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_0_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_0_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_1_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_2_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_3_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_4_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_5_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_6_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_7_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_8_bias_new complete dim = 1

  do_compute2(in, out, reps);
}
