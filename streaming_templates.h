
///////////////////////////////////////////////////////////
/// Author Yun Feng(fengyun@usc.edu)
///        Arash Fayyazi(fayyazi@usc.edu)
///        Amirhossein Esmaili Dastjerdi(esmailid@usc.edu)
/// Date 04/12/2023
/// Org USC
////////////////////////////////////////////////////////////
#include "bnn-library.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "dma.h"
#include "mvau.hpp"
#include "conv.hpp"
#include "hls_half.h"

#ifndef __StreamingTemplates_H__FG
#define __StreamingTemplates_H__

using namespace hls;

#define max_op(a,b) ((a)>(b)?(a):(b))

// Function for simd > 32, reads filter from buffer to stream
// Parameters:
// buffer_weights: A pointer to the buffer where the actual data is stored.
// out_stream: A pointer to the stream where the data will be sent.
// startIdx: The starting index of buffer_weights for the current layer.
// endIdx1 and endIdx2: The ending indices of buffer_weights for the current layer.
// pe: The processing elements count.
// left_pe: The count of the remaining processing elements.
// simd_r: The SIMD rate.
// ofm: The output feature map count.
template <int DATA_WIDTH, int SIMD_W, int startIdx, int endIdx1, int endIdx2, int pe, int left_pe, int simd_r, int ofm>
	void read_weight_long(ap_uint<DATA_WIDTH * SIMD_W> *buffer_weights, stream<ap_uint<DATA_WIDTH * SIMD_W>> *out_stream)
	{
		cout << "Reading weights (long)" << endl;

		// Temp storage for weights
		ap_uint<DATA_WIDTH * SIMD_W> temp_weights[pe *simd_r];

		// Loop through each of the output feature maps
		for (int i = 0; i < ofm * ofm; i++)
		{
			// Loop through the weights from startIdx to endIdx1
			for (int j = startIdx; j < endIdx1; j = j + pe *simd_r)
			{
				for (int k = 0; k < pe; k++)
				{
					for (int l = 0; l < simd_r; l++)
					{
					#pragma HLS PIPELINE II = 1
						temp_weights[k *simd_r + l] = buffer_weights[j + k *simd_r + l];
						out_stream[k *simd_r + l].write(temp_weights[k *simd_r + l]);
					}
				}
			}

			// Loop through the weights from endIdx1 to endIdx2
			for (int j = endIdx1; j < endIdx2; j = j + left_pe *simd_r)
			{
				for (int k = 0; k < left_pe; k++)
				{
					for (int l = 0; l < simd_r; l++)
					{
					#pragma HLS PIPELINE II = 1
						temp_weights[k *simd_r + l] = buffer_weights[j + k *simd_r + l];
						out_stream[k *simd_r + l].write(temp_weights[k *simd_r + l]);
					}
				}
			}
		}
	}

// Function for simd <= 32, reads filter from buffer to stream
// Function for simd <= 32, reads filter from buffer to stream
// Parameters are the same as read_weight_long, with an additional T which represents simd*WEIGHT_WIDTH that determines each 
// element's data width.

template <int DATA_WIDTH, int SIMD_W, int T, int startIdx, int endIdx1, int endIdx2, int pe, int left_pe, int ofm>
	void read_weight_short(ap_uint<DATA_WIDTH * SIMD_W> *buffer_weights, stream<ap_uint<T>> *out_stream)
	{
		cout << "Reading weights (short)" << endl;

		// Temp storage for weights
		ap_uint<T> temp_weights[pe];

		// Loop through each of the output feature maps
		for (int i = 0; i < ofm * ofm; i++)
		{
			// Loop through the weights from startIdx to endIdx1
			for (int j = startIdx; j < endIdx1; j = j + pe)
			{
				for (int k = 0; k < pe; k++)
				{
				#pragma HLS PIPELINE II = 1
					temp_weights[k] = buffer_weights[j + k](T - 1, 0);
					out_stream[k].write(temp_weights[k]);
				}
			}

			// Loop through the weights from endIdx1 to endIdx2
			for (int j = endIdx1; j < endIdx2; j = j + left_pe)
			{
				for (int k = 0; k < left_pe; k++)
				{
				#pragma HLS PIPELINE II = 1
					temp_weights[k] = buffer_weights[j + k](T - 1, 0);
					out_stream[k].write(temp_weights[k]);
				}
			}
		}
	}

// Function for simd > 32, reads input data from buffer to stream
// Parameters:
// buffer_IFM: A pointer to the buffer where the actual data is stored.
// out_stream: The stream where the data will be sent.
// simd: The SIMD value, which represents parallelism in the channel direction.
// IFMDim: The width and height of the input feature map.
// IFMCha: The channel count of the input feature map.
template <int DATA_WIDTH, int simd, int IFMDim, int IFMCha>
	void read_in_long(ap_uint<512> *buffer_IFM, stream<ap_uint<512>> &out_stream)
	{
		// Loop through the input feature map data
		for (int i = 0; i < ((IFMDim) *(IFMDim) *(IFMCha)) / simd; i++)
		{
			// Write the data to the output stream
			out_stream.write(buffer_IFM[i]);
		}
	}

// Function for simd <= 32, reads input data from buffer to stream
// Parameters are the same as read_in_long, with an additional T which represents each element's data width.
template <int DATA_WIDTH, int T, int simd, int IFMDim, int IFMCha>
	void read_in_short(ap_uint<512> *buffer_IFM, stream<ap_uint<T>> &out_stream)
	{
		cout << "Reading IFM" << endl;

		// Loop through the input feature map data
		for (int i = 0; i < ((IFMDim) *(IFMDim) *(IFMCha)) / simd; i++)
		{
			// Write the data to the output stream
			out_stream.write(buffer_IFM[i](simd *DATA_WIDTH - 1, 0));
		}
	}

/////////
template < int T, int simd, int IFMDim, int IFMCha, int in_width>
	void read_and_normalize(ap_uint<8> *buffer_IFM, stream<ap_uint<T>> &out_stream)
	{
		cout << "Reading and processing IFM" << endl;

		// Loop through the input feature map data
		ap_uint<T> tmp;
		int M[3] = {0,1,2};
		int S[3] = {0,1,2};
		for (int i = 0; i < IFMDim *IFMDim; i++)
		{
			for (int j = 0; j < IFMCha/simd; j++) {
				for (int k = 0; k < simd; k++) {
					//int tmp_idx = (j * simd + k);
					//int buff_idx = (i * IFMCha/simd + tmp_idx);
                    #pragma HLS PIPELINE II = 1
					tmp(((j * simd + k) + 1) * in_width - 1,  (j * simd + k) * in_width) = (buffer_IFM[(i * IFMCha/simd + (j * simd + k))] + M[(j * simd + k)]) * S[(j * simd + k)];
				}
			}
			out_stream.write(tmp);
		}
	}

// Function for writing results from the input stream to a buffer
// Parameters:
// input_stream: The input stream from where the data will be read.
// output_buffer: A pointer to the buffer where the result will be written.
template < int size>
	void write_result(stream<ap_uint<512> > &input_stream, ap_uint<512> *output_buffer)
	{
		// Loop through the input stream data
		for (int i = 0; i < 2048; i++)
		{
			// Write the data to the output buffer
			output_buffer[i] = input_stream.read();
		}
	}

/*
 *This function generates reused data for SIMD > 32 based on the input stream and each layer's info.
 *
 *Parameters:
 *IFM: Stream where the actual data is stored
 *out_stream: Stream where the data will be sent
 *simd_r: For SIMD larger than 32, this determines how much data should be read for the entire SIMD. For example, for SIMD=64, simd_r=2
 *simd_loop: For some cases like channel=512, SIMD=128, there should be 4 loops to go through the entire channel
 *IFM_p: Pre-stored input feature map columns
 *KerDim: Kernel width and height
 *pe: PE number
 *IFMDim: IFM width and height
 *OFM_s: OFM width and height
 *pe_loop_times: Determines how many times PE should loop to cover all kernels. For example, if we have 16 kernels and PE=3, then pe_loop_times=6
 *left_pe: Determines the last PE loop's PE number. For example, if we have 16 kernels and PE=3, then left_pe=1. If 15 kernels and PE=3, then left_pe=3
 *start_point and end_point: Determine how many elements should be read out from the weight stream
 */
template <int DATA_WIDTH, int SIMD_I, int simd_r, int simd_loop, int IFM_p, int KerDim, int pe, int IFMDim, int OFM_s, int pe_loop_times, int left_pe, int start_point, int end_point>
	void Input_Generator_long(stream<ap_uint<DATA_WIDTH * SIMD_I>> &IFM, stream<ap_uint<DATA_WIDTH * SIMD_I>> *out_stream)
	{
		cout << "Input Generator long" << endl;
		ap_uint<DATA_WIDTH * SIMD_I> arr[IFMDim *IFM_p][simd_r *simd_loop];
		#pragma HLS array_partition variable = arr complete dim = 2
		//#pragma HLS ARRAY_PARTITION variable=arr uram
		#pragma HLS bind_storage variable = arr type = RAM_1P impl = uram
		int head = 0;
		int tail = 0;
		int distance = 0;

		int index_tail_move = 0;
		int outer_loop = 0;

		ap_uint<1> tail_end = 0;
		ap_uint<1> full_enable = 0;

		ap_uint<DATA_WIDTH * SIMD_I> tmp_out_0[simd_r *simd_loop];
		ap_uint<DATA_WIDTH * SIMD_I> tmp_out_1[simd_r *simd_loop];
		//deal with the distance value
		loop1:
			while (outer_loop < (end_point - start_point))
			{
				//base on the number of weights that go through fifo determine the IFM loop-times
				// Calculate distance value
				int tail_head = tail - head;
				if (tail_head > 0)
					distance = tail_head;
				else if (tail_head < 0)
					distance = tail_head + IFM_p;
				else if (full_enable == 0)
					distance = 0;
				else if (full_enable == 1)
					distance = IFM_p;

				//based on distance determine whether read into arr or write out to stream
				if (distance >= KerDim)
				{
					//write stream part(from arr to output)
					full_enable = 1;
					//start to write out to output stream
					loop2:
						for (int i = 0; i < OFM_s; i++)
						{
							//IFM high dimension
							loop3: for (int j = 0; j < pe_loop_times; j++)
							{
								//move to following pe filters
								loop4: for (int g = 0; g < simd_loop; g++)
								{
									for (int m = 0; m < KerDim; m++)
									{
										//filter width dimension
										loop5: for (int n = 0; n < KerDim; n++)
										{
											//filter high dimension
											int tmp;
											if ((m + head) >= IFM_p)
												tmp = m + head - IFM_p;
											else
												tmp = m + head;

											if (j < (pe_loop_times - 1))
											{
												//for some layers, filter number cannot be divided by pe, hence for the front filters use pe as loop times, for the remain filters use left_pe as loop times

												loop6: for (int L = 0; L < simd_r; L++)
												{
												#pragma HLS UNROLL
													tmp_out_0[L] = arr[tmp *IFMDim + n + i][g *simd_r + L];
													loop7:
														for (int k = 0; k < pe; k++)
														{
															//go over each pe filters
															#pragma HLS UNROLL
															out_stream[k *simd_r + L].write(tmp_out_0[L]);
															//debug2<<tmp_out_0[L]<<endl;
															outer_loop++;
														}
												}
											}
											else
											{
												loop8: for (int L = 0; L < simd_r; L++)
												{
													#pragma HLS UNROLL
													tmp_out_1[L] = arr[tmp *IFMDim + n + i][g *simd_r + L];
													loop9:
														for (int k = 0; k < left_pe; k++)
														{
															//go over each pe filters
															#pragma HLS UNROLL
															out_stream[k *simd_r + L].write(tmp_out_1[L]);
															//debug2<<tmp_out_1[L]<<endl;
															outer_loop++;
														}
												}
											}
										}
									}
								}
							}
						}

					head++;
					if (head == IFM_p)
					{
						//store the head pointer address
						head = 0;
					}
				}

				if (distance < IFM_p && index_tail_move < IFMDim)
				{
					//read in IFM part(to arr)
					if (tail == IFM_p)
					{
						//store the tail pointer address
						tail = 0;
					}

					loop10:
						for (int i = 0; i < IFMDim; i++)
						{
							loop11: for (int j = 0; j < simd_r * simd_loop; j++)
							{
							#pragma HLS PIPELINE II = 1
								arr[tail *IFMDim + i][j] = IFM.read();
							}
						}

					index_tail_move++;	//store the tail index move operation times

					tail++;
				}
			}
	}

//function -> for simd<=32, based on input stream and each layer's info generate reused data and put it to a stream
//IN_T-> input stream width
//IN_row-> the entire IFM channels occupy rows in the stream
//IFM -> where the actual data being stored
//out_stream -> where the data go
//T -> determine the read in and write out stream data width
//IFM_p -> pre-store input feature map columns
//KerDim -> kernel width and high
//pe -> pe number
//IFMDim -> IFM width and high
//OFM_s -> OFM width and high
//pe_loop_times -> determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//start_point and end_point are as same as the read_weight(), used to determine how many elements should be read out from weight stream
template <int DATA_WIDTH, int SIMD_I, int IN_T, int IN_row, int T, int IFM_p, int KerDim, int pe, int IFMDim, int OFM_s, int pe_loop_times, int left_pe, int start_point, int end_point>
	void Input_Generator_short_1(stream<ap_uint<IN_T> > &IFM, stream<ap_uint<T>> *out_stream)
	{
		cout << "Input_Generator_short_1" << endl;
		ap_uint<IN_T> arr[IN_row *IFMDim *IFM_p];
		//#pragma HLS ARRAY_PARTITION variable=arr complete dim=0
		//#pragma HLS ARRAY_PARTITION variable=arr uram
		
		#pragma HLS bind_storage variable = arr type = RAM_1P impl = uram

		int head = 0;
		int tail = 0;
		int distance = 0;

		int index_tail_move = 0;
		int outer_loop = 0;

		ap_uint<1> tail_end = 0;
		ap_uint<1> full_enable = 0;

		//deal with the distance value
		loop1:
			while (outer_loop < (end_point - start_point))
			{
				//base on the number of weights that go through fifo determine the IFM loop-times
				//cout<<"here we are"<<endl;
				int tail_head = tail - head;
				if (tail_head > 0)
				{
					distance = tail_head;
				}

				if (tail_head < 0)
				{
					distance = tail_head + IFM_p;
				}

				if (full_enable == 0 && (tail_head == 0))
				{
					distance = 0;
				}

				if (full_enable == 1 && (tail_head == 0))
				{
					distance = IFM_p;
				}

				//based on distance determine whether read into arr or write out to stream
				if (distance >= KerDim)
				{
					//write stream part(from arr to output)
					full_enable = 1;
					//start to write out to output stream
					loop2:
						for (int i = 0; i < OFM_s; i++)
						{
							//IFM high dimension
							loop3: for (int j = 0; j < pe_loop_times; j++)
							{
								//move to following pe filters
								loop4: for (int m = 0; m < KerDim; m++)
								{
									//filter width dimension
									loop5: for (int n = 0; n < KerDim; n++)
									{
										//filter high dimension
										int tmp;
										if ((m + head) >= IFM_p)
											tmp = m + head - IFM_p;
										else
											tmp = m + head;

										for (int g = 0; g < IN_row; g++)
										{
											ap_uint<IN_T> tmp_out = arr[tmp *IFMDim *IN_row + (n + i) *IN_row + g];
											if (j < (pe_loop_times - 1))
											{
												//for some layers, filter number cannot be divided by pe, hence for the front filters use pe as loop times, for the remain filters use left_pe as loop times
												loop6: for (int k = 0; k < pe; k++)
												{
													//go over each pe filters
													#pragma HLS UNROLL
													for (int a = 0; a < IN_T / T; a++)
													{
														out_stream[k].write(tmp_out((a + 1) *T - 1, a *T));
														//debug3<<tmp_out<<endl;
														outer_loop++;
													}
												}
											}
											else
											{
												loop7: for (int k = 0; k < left_pe; k++)
												{
													//go over each pe filters
													#pragma HLS UNROLL
													for (int a = 0; a < IN_T / T; a++)
													{
														out_stream[k].write(tmp_out((a + 1) *T - 1, a *T));
														//debug3<<tmp_out<<endl;
														outer_loop++;
													}
												}
											}
										}
									}
								}
							}
						}

					head++;
					if (head == IFM_p)
					{
						//store the head pointer address
						head = 0;
					}
				}

				if (distance < IFM_p && index_tail_move < IFMDim)
				{
					//read in IFM part(to arr)
					if (tail == IFM_p)
					{
						//store the tail pointer address
						tail = 0;
					}

					loop8:
						for (int i = 0; i < IFMDim; i++)
						{
							for (int j = 0; j < IN_row; j++)
							{
							#pragma HLS PIPELINE II = 1
								arr[tail *IFMDim *IN_row + i *IN_row + j] = IFM.read();
								//debug2<<arr[tail*IFMDim + i]<<endl;
							}
						}

					index_tail_move++;	//store the tail index move operation times

					tail++;
				}
			}
	}

//function -> for simd<=32, based on input stream and each layer's info generate reused data and put it to a stream
//IN_T-> input stream width
//IN_row-> the entire IFM channels occupy rows in the stream
//IFM -> where the actual data being stored
//out_stream -> where the data go
//T -> determine the read in and write out stream data width
//IFM_p -> pre-store input feature map columns
//KerDim -> kernel width and high
//pe -> pe number
//IFMDim -> IFM width and high
//OFM_s -> OFM width and high
//pe_loop_times -> determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//start_point and end_point are as same as the read_weight(), used to determine how many elements should be read out from weight stream
template <int DATA_WIDTH, int SIMD_I, int IN_row, int IN_T, int T, int IFM_p, int KerDim, int pe, int IFMDim, int OFM_s, int pe_loop_times, int left_pe, int start_point, int end_point>
	void Input_Generator_short_2(stream<ap_uint<IN_T> > &IFM, stream<ap_uint<T>> *out_stream)
	{
		cout << "Input_Generator_short_2" << endl;
		ap_uint<IN_T> arr[IN_row *IFMDim *IFM_p];
		//#pragma HLS ARRAY_PARTITION variable=arr complete dim=0
		//#pragma HLS ARRAY_PARTITION variable=arr uram
		
		#pragma HLS bind_storage variable = arr type = RAM_1P impl = uram

		int head = 0;
		int tail = 0;
		int distance = 0;

		int index_tail_move = 0;
		int outer_loop = 0;

		ap_uint<1> tail_end = 0;
		ap_uint<1> full_enable = 0;

		//deal with the distance value
		loop1:
			while (outer_loop < (end_point - start_point))
			{
				//base on the number of weights that go through fifo determine the IFM loop-times
				//cout<<"here we are"<<endl;
				int tail_head = tail - head;
				if (tail_head > 0)
				{
					distance = tail_head;
				}

				if (tail_head < 0)
				{
					distance = tail_head + IFM_p;
				}

				if (full_enable == 0 && (tail_head == 0))
				{
					distance = 0;
				}

				if (full_enable == 1 && (tail_head == 0))
				{
					distance = IFM_p;
				}

				//based on distance determine whether read into arr or write out to stream
				if (distance >= KerDim)
				{
					//write stream part(from arr to output)
					full_enable = 1;
					//start to write out to output stream
					loop2:
						for (int i = 0; i < OFM_s; i++)
						{
							//IFM high dimension
							//#pragma HLS loop_flatten off
							loop3: for (int j = 0; j < pe_loop_times; j++)
							{
								//move to following pe filters
								loop4: for (int m = 0; m < KerDim; m++)
								{
									//filter width dimension
									loop5: int tmp;
									if ((m + head) >= IFM_p)
										tmp = m + head - IFM_p;
									else
										tmp = m + head;
									for (int n = 0; n < KerDim; n++)
									{
										//filter high dimension

										for (int g = 0; g < IN_row; g++)
										{
											ap_uint<IN_T> tmp_out = arr[tmp *IFMDim *IN_row + (n + i) *IN_row + g];
											if (j < (pe_loop_times - 1))
											{
												//for some layers, filter number cannot be divided by pe, hence for the front filters use pe as loop times, for the remain filters use left_pe as loop times
												loop6: for (int k = 0; k < pe; k++)
												{
													//go over each pe filters
													#pragma HLS UNROLL
													for (int a = 0; a < IN_T / T; a++)
													{
														out_stream[k].write(tmp_out((a + 1) *T - 1, a *T));
														//debug3<<tmp_out<<endl;
														outer_loop++;
													}
												}
											}
											else
											{
												loop7: for (int k = 0; k < left_pe; k++)
												{
													//go over each pe filters
													#pragma HLS UNROLL
													for (int a = 0; a < IN_T / T; a++)
													{
														out_stream[k].write(tmp_out((a + 1) *T - 1, a *T));
														//debug3<<tmp_out<<endl;
														outer_loop++;
													}
												}
											}
										}
									}
								}
							}
						}

					head++;
					if (head == IFM_p)
					{
						//store the head pointer address
						head = 0;
					}
				}

				if (distance < IFM_p && index_tail_move < IFMDim)
				{
					//read in IFM part(to arr)
					if (tail == IFM_p)
					{
						//store the tail pointer address
						tail = 0;
					}

					loop8:
						for (int i = 0; i < IFMDim; i++)
						{
							for (int j = 0; j < IN_row; j++)
							{
							#pragma HLS PIPELINE II = 1
								arr[tail *IFMDim *IN_row + i *IN_row + j] = IFM.read();
								//debug2<<arr[tail*IFMDim + i]<<endl;
							}
						}

					index_tail_move++;	//store the tail index move operation times

					tail++;
				}
			}
	}

//function -> for simd>32, Mac filters with IFM by stream format
//IFM -> IFM data stream
//WEIGHT -> WEIGHT data stream
//out_stream -> where the data go
//IFM_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then IFM_r=2
//WEIGHT_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then WEIGHT_r=2
//pe -> pe number
//KerDim -> Kernel row and column number
//OFMDim -> OFM row and column number
//pe_loop_times -> pe window slide times, determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//simd_loop_times -> determines the number of simd we need to use to calculate entire channel. Eg, channel=256, simd=64, then we need to move simd 4 times, then simd_loop_times=4
template <typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int IFM_r, int WEIGHT_r, int pe, int KerDim, int OFMDim,
	int pe_loop_times, int left_pe, int simd_loop_times >
	void Mac_long(stream<ap_uint<DATA_WIDTH * SIMD_I>> *IFM, stream<ap_uint<DATA_WIDTH * SIMD_W>> *WEIGHT, stream<ap_uint<DATA_WIDTH * SIMD_I_NEXT>> &output)
	{
		cout << "Mac_long" << endl;
		T_data in[pe][IFM_r][SIMD_W];
		T_mac tmp[pe];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[pe][IFM_r][SIMD_W];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[pe][IFM_r][SIMD_W];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<DATA_WIDTH * SIMD_I> tmp_i[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<DATA_WIDTH * SIMD_W> tmp_w[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column
			loop1: for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result
				loop2:
					for (int p = 0; p < pe_loop_times; p++)
					{
						//pe window shift
						for (int f = 0; f < pe; f++)
						{
						#pragma HLS UNROLL
							tmp[f] = 0;
						}

						loop3:
							for (int s = 0; s < simd_loop_times; s++)
							{
								//for some cases channel number is way larger than simd, them need multiple simd to calculate all channel. Eg. channel num = 64, simd=32, then need 2 loops
								loop4: for (int l = 0; l < KerDim; l++)
								{
									//traverse weight column
									//#pragma HLS PIPELINE OFF
									loop5: for (int m = 0; m < KerDim; m++)
									{
										//traverse weight row
										#pragma HLS PIPELINE
										loop6:
											for (int k = 0; k < pe; k++)
											{
												//pe pipeline
												#pragma HLS UNROLL
												if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
													break;
												loop7:
													for (int r = 0; r < IFM_r; r++)
													{
													#pragma HLS UNROLL
														tmp_i[k][r] = IFM[k *IFM_r + r].read();
														tmp_w[k][r] = WEIGHT[k *IFM_r + r].read();
														loop8:
															for (int n = 0; n < SIMD_W; n++)
															{
																//simd parallel
																#pragma HLS UNROLL

																	in[k][r][n].range() = tmp_i[k][r].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
																wt[k][r][n].range() = tmp_w[k][r].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);

																tmp_[k][r][n] = in[k][r][n] *wt[k][r][n];
																tmp[k] += tmp_[k][r][n];
																//debug3<<(tmp_i[r]((n+1)*DATA_WIDTH-1,n*DATA_WIDTH))<<"*"<<(tmp_w[r]((n+1)*WEIGHT_WIDTH-1,n*WEIGHT_WIDTH))<<endl;
															}
													}
											}
									}
								}
							}

						loop9:
							for (int r = 0; r < pe; r++)
							{
								//traverse all tmp result
								#pragma HLS PIPELINE II = 1
								T_mac tmp_relu = max_op(tmp[r], (T_mac) 0); 
								result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
								//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
								if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
								{
									//if this is the last round and this round is end, write out the result to output directly
									output.write(result);
									//debug3<<result<<endl;
									break;
								}

								if (index == SIMD_I_NEXT - 1)
								{
									//if result is full, write to output and initialize result's value and index's value
									output.write(result);
									//debug2<<result<<endl;
									result = 0;
									index = 0;
								}
								else
								{
									index++;	//increase index every time it less than SIMD_I_TH-1
								}
							}
					}
			}
		}
	}

//function -> for simd>32, Mac filters with IFM by stream format
//IFM -> IFM data stream
//WEIGHT -> WEIGHT data stream
//out_stream -> where the data go
//IFM_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then IFM_r=2
//WEIGHT_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then WEIGHT_r=2
//pe -> pe number
//KerDim -> Kernel row and column number
//OFMDim -> OFM row and column number
//pe_loop_times -> pe window slide times, determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//simd_loop_times -> determines the number of simd we need to use to calculate entire channel. Eg, channel=256, simd=64, then we need to move simd 4 times, then simd_loop_times=4
template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int SIMD_I_NEXT_2, int IFM_r, int WEIGHT_r, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_long_2out(stream<ap_uint<DATA_WIDTH * SIMD_I> > *IFM, stream<ap_uint<DATA_WIDTH * SIMD_W> > *WEIGHT, stream<ap_uint<DATA_WIDTH * SIMD_I_NEXT>> &output,
		stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT_2>> &output_2)
	{
		cout << "Mac_long" << endl;
		T_data in[pe][IFM_r][SIMD_W];
		T_mac tmp[pe];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[pe][IFM_r][SIMD_W];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[pe][IFM_r][SIMD_W];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<DATA_WIDTH * SIMD_I> tmp_i[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<DATA_WIDTH * SIMD_W> tmp_w[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column
			loop1: for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				ap_uint<DATA_WIDTH * SIMD_I_NEXT_2> result_2 = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result
				int index_2 = 0;	//record the number of channels than written to result
				loop2:
					for (int p = 0; p < pe_loop_times; p++)
					{
						//pe window shift
						for (int f = 0; f < pe; f++)
						{
						#pragma HLS UNROLL
							tmp[f] = 0;
						}

						loop3:
							for (int s = 0; s < simd_loop_times; s++)
							{
								//for some cases channel number is way larger than simd, them need multiple simd to calculate all channel. Eg. channel num = 64, simd=32, then need 2 loops
								loop4: for (int l = 0; l < KerDim; l++)
								{
									//traverse weight column
									//#pragma HLS PIPELINE OFF
									loop5: for (int m = 0; m < KerDim; m++)
									{
										//traverse weight row
										#pragma HLS PIPELINE
										loop6:
											for (int k = 0; k < pe; k++)
											{
												//pe pipeline
												#pragma HLS UNROLL
												if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
													break;
												loop7:
													for (int r = 0; r < IFM_r; r++){
														#pragma HLS UNROLL
														tmp_i[k][r] = IFM[k *IFM_r + r].read();
														tmp_w[k][r] = WEIGHT[k *IFM_r + r].read();
														loop8:
															for (int n = 0; n < SIMD_W; n++)
															{
																//simd parallel
																#pragma HLS UNROLL

																	in[k][r][n].range() = tmp_i[k][r].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
																wt[k][r][n].range() = tmp_w[k][r].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);

																tmp_[k][r][n] = in[k][r][n] *wt[k][r][n];
																tmp[k] += tmp_[k][r][n];
																//debug3<<(tmp_i[r]((n+1)*DATA_WIDTH-1,n*DATA_WIDTH))<<"*"<<(tmp_w[r]((n+1)*WEIGHT_WIDTH-1,n*WEIGHT_WIDTH))<<endl;
															}
													}
											}
									}
								}
							}

						loop9:
							for (int r = 0; r < pe; r++)
							{
								//traverse all tmp result							
								#pragma HLS PIPELINE II = 1
								T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
								result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
								result_2((index_2 + 1) *DATA_WIDTH - 1, index_2 *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
								//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
								if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
								{
									//if this is the last round and this round is end, write out the result to output directly
									output.write(result);
									output_2.write(result_2);
									//debug3<<result<<endl;
									break;
								}

								if (index == SIMD_I_NEXT - 1)
								{
									//if result is full, write to output and initialize result's value and index's value
									output.write(result);
									//debug2<<result<<endl;
									result = 0;
									index = 0;
								}
								else
								{
									index++;	//increase index every time it less than SIMD_I_TH-1
								}
								if (index_2 == SIMD_I_NEXT_2 - 1)
								{
									//if result is full, write to output and initialize result's value and index's value
									output_2.write(result_2);
									//debug2<<result<<endl;
									result_2 = 0;
									index_2 = 0;
								}
								else
								{
									index_2++;	//increase index every time it less than SIMD_I_TH-1
								}
							}
					}
			}
		}
	}

//function -> for simd<=32, Mac filters with IFM by stream format
//IFM -> IFM data stream
//WEIGHT -> WEIGHT data stream
//out_stream -> where the data go
//IFM_size -> determine IFM data width, related to simd*DATA_WIDTH
//WEIGHT_size -> determine WEIGHT data width, related to simd*WEIGHT_WIDTH
//simd -> simd number
//pe -> pe number
//KerDim -> Kernel row and column number
//OFMDim -> OFM row and column number
//pe_loop_times -> pe window slide times, determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//simd_loop_times -> determines the number of simd we need to use to calculate entire channel. Eg, channel=256, simd=64, then we need to move simd 4 times, then simd_loop_times=4
template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_1(stream<ap_uint<IFM_size>> *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &output)
	{
		cout << "Mac short_1" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					 			//#pragma HLS PIPELINE
						loop1: for (int l = 0; l < KerDim; l++)
						{
							//traverse weight column
							//#pragma HLS PIPELINE
							#pragma HLS loop_flatten off
							loop2:
								for (int m = 0; m < KerDim; m++)
								{
									//traverse weight row
									#pragma HLS PIPELINE
									loop3:
										for (int k = 0; k < pe; k++)
										{
										#pragma HLS UNROLL
											if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
												break;

											tmp_i[k] = IFM[k].read();

											tmp_w[k] = WEIGHT[k].read();
											loop4:
												for (int n = 0; n < simd; n++)
												{
													//simd parallel
													#pragma HLS UNROLL
														in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
													wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
													//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
													tmp_[n] = in[n] *wt[n];
													tmp[k] += tmp_[n];
													//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
												}
										}
								}
						}
					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
						//debug2<<tmp[r]<<endl;
						if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
						{
							//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							//debug3<<result<<endl;
							break;
						}

						if (index == SIMD_I_NEXT - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug3<<result<<endl;
							result = 0;
							index = 0;
						}
						else
						{
							index++;	//increase index every time it less than SIMD_I_TH-1

						}

						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}

template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_2(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &output)
	{
		cout << "Mac short_2" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					//#pragma HLS PIPELINE
						loop1:
							for (int l = 0; l < KerDim; l++)
							{
								//traverse weight column
								//#pragma HLS PIPELINE
								loop2:
									for (int m = 0; m < KerDim; m++)
									{
										//traverse weight row
										#pragma HLS PIPELINE
										loop3:
											for (int k = 0; k < pe; k++)
											{
											#pragma HLS UNROLL
												if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
													break;

												tmp_i[k] = IFM[k].read();

												tmp_w[k] = WEIGHT[k].read();
												loop4:
													for (int n = 0; n < simd; n++)
													{
														//simd parallel
														#pragma HLS UNROLL
															in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
														wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
														//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
														tmp_[n] = in[n] *wt[n];
														tmp[k] += tmp_[n];
														//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
													}
											}
									}
							}
					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
						//debug2<<tmp[r]<<endl;
						if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
						{
							//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							//debug3<<result<<endl;
							break;
						}

						if (index == SIMD_I_NEXT - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug3<<result<<endl;
							result = 0;
							index = 0;
						}
						else
						{
							index++;	//increase index every time it less than SIMD_I_TH-1

						}

						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}

template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_2_less_loops(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &output)
	{
		cout << "Mac short_2" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					#pragma HLS PIPELINE

						loop3:
							for (int k = 0; k < pe; k++)
							{
							#pragma HLS UNROLL
								if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
									break;

								tmp_i[k] = IFM[k].read();

								tmp_w[k] = WEIGHT[k].read();
								loop4:
									for (int n = 0; n < simd; n++)
									{
										//simd parallel
										#pragma HLS UNROLL
											in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
										wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
										//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
										tmp_[n] = in[n] *wt[n];
										tmp[k] += tmp_[n];
										//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
									}
							}

					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
						//debug2<<tmp[r]<<endl;
						if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
						{
							//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							//debug3<<result<<endl;
							break;
						}

						if (index == SIMD_I_NEXT - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug3<<result<<endl;
							result = 0;
							index = 0;
						}
						else
						{
							index++;	//increase index every time it less than SIMD_I_TH-1

						}

						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}
	
	
template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_out(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<half> &output)
	{
		cout << "Mac short_out" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				half result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					//#pragma HLS PIPELINE
						loop1:
							for (int l = 0; l < KerDim; l++)
							{
								//traverse weight column
								//#pragma HLS PIPELINE
								loop2:
									for (int m = 0; m < KerDim; m++)
									{
										//traverse weight row
										#pragma HLS PIPELINE
										loop3:
											for (int k = 0; k < pe; k++)
											{
											#pragma HLS UNROLL
												if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
													break;

												tmp_i[k] = IFM[k].read();

												tmp_w[k] = WEIGHT[k].read();
												loop4:
													for (int n = 0; n < simd; n++)
													{
														//simd parallel
														#pragma HLS UNROLL
															in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
														wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
														//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
														tmp_[n] = in[n] *wt[n];
														tmp[k] += tmp_[n];
														//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
													}
											}
									}
							}
					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result = (half) tmp_relu;
						output.write(result);


						//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;


						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}

template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int SIMD_I_NEXT_2, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_2_2out(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &output,
		stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT_2>> &output_2)
	{
		cout << "Mac short_2" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result
				ap_uint<DATA_WIDTH * SIMD_I_NEXT_2> result_2 = 0;	//in case output feature map channel larger than 512bits
				int index_2 = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					//#pragma HLS PIPELINE
						loop1:
							for (int l = 0; l < KerDim; l++)
							{
								//traverse weight column
								//#pragma HLS PIPELINE
								loop2:
									for (int m = 0; m < KerDim; m++)
									{
										//traverse weight row
										#pragma HLS PIPELINE
										loop3:
											for (int k = 0; k < pe; k++)
											{
											#pragma HLS UNROLL
												if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
													break;

												tmp_i[k] = IFM[k].read();

												tmp_w[k] = WEIGHT[k].read();
												loop4:
													for (int n = 0; n < simd; n++)
													{
														//simd parallel
														#pragma HLS UNROLL
															in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
														wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
														//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
														tmp_[n] = in[n] *wt[n];
														tmp[k] += tmp_[n];
														//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
													}
											}
									}
							}
					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						result_2((index_2 + 1) *DATA_WIDTH - 1, index_2 *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
						//debug2<<tmp[r]<<endl;
						if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
						{
							//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							output_2.write(result_2);
							//debug3<<result<<endl;
							break;
						}

						if (index == SIMD_I_NEXT - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug3<<result<<endl;
							result = 0;
							index = 0;
						}
						else
						{
							index++;	//increase index every time it less than SIMD_I_TH-1

						}
						if (index_2 == SIMD_I_NEXT_2 - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output_2.write(result_2);
							//debug3<<result<<endl;
							result_2 = 0;
							index_2 = 0;
						}
						else
						{
							index_2++;	//increase index every time it less than SIMD_I_TH-1

						}

						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}


template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int SIMD_I_NEXT, int SIMD_I_NEXT_2, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_2_2out_less_loops(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &output,
		stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT_2>> &output_2)
	{
		cout << "Mac short_2" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0

		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column

			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				ap_uint<DATA_WIDTH * SIMD_I_NEXT> result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result
				ap_uint<DATA_WIDTH * SIMD_I_NEXT_2> result_2 = 0;	//in case output feature map channel larger than 512bits
				int index_2 = 0;	//record the number of channels than written to result

				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}

					for (int s = 0; s < simd_loop_times; s++)
					{
					#pragma HLS PIPELINE
					
					loop3:
						for (int k = 0; k < pe; k++)
						{
						#pragma HLS UNROLL
							if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
								break;
					
							tmp_i[k] = IFM[k].read();
					
							tmp_w[k] = WEIGHT[k].read();
							loop4:
								for (int n = 0; n < simd; n++)
								{
									//simd parallel
									#pragma HLS UNROLL
										in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
									wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
									//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
									tmp_[n] = in[n] *wt[n];
									tmp[k] += tmp_[n];
									//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
								}
						}
					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result((index + 1) *DATA_WIDTH - 1, index *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						result_2((index_2 + 1) *DATA_WIDTH - 1, index_2 *DATA_WIDTH) = tmp_relu.range(DATA_WIDTH - 1, 0);	//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;
						//debug2<<tmp[r]<<endl;
						if ((p == (pe_loop_times - 1)) && (r == (left_pe - 1)))
						{
							//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							output_2.write(result_2);
							//debug3<<result<<endl;
							break;
						}

						if (index == SIMD_I_NEXT - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug3<<result<<endl;
							result = 0;
							index = 0;
						}
						else
						{
							index++;	//increase index every time it less than SIMD_I_TH-1

						}
						if (index_2 == SIMD_I_NEXT_2 - 1)
						{
							//if result is full, write to output and initialize result's value and index's value
							output_2.write(result_2);
							//debug3<<result<<endl;
							result_2 = 0;
							index_2 = 0;
						}
						else
						{
							index_2++;	//increase index every time it less than SIMD_I_TH-1

						}

						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}
	
template < typename T_data, typename T_weight, typename T_mul, typename T_mac, int DATA_WIDTH, int WEIGHT_WIDTH, int SIMD_I, int SIMD_W, int IFM_size, int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times, int left_pe, int simd_loop_times>
	void Mac_short_out_less_loops(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT, stream<half> &output)
	{
		cout << "Mac short_out" << endl;
		T_data in[simd];
		#pragma HLS ARRAY_PARTITION variable = in complete dim = 0
		T_weight wt[simd];
		#pragma HLS ARRAY_PARTITION variable = wt complete dim = 0
		T_mul tmp_[simd];
		#pragma HLS ARRAY_PARTITION variable = tmp_ complete dim = 0

		ap_uint<IFM_size> tmp_i[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_i complete dim = 0
		ap_uint<WEIGHT_size> tmp_w[pe];
		#pragma HLS ARRAY_PARTITION variable = tmp_w complete dim = 0
		loop1:
		for (int i = 0; i < OFMDim; i++)
		{
			//IFM column
			loop2:
			for (int j = 0; j < OFMDim; j++)
			{
				//IFM row
				half result = 0;	//in case output feature map channel larger than 512bits
				int index = 0;	//record the number of channels than written to result
				loop3:
				for (int p = 0; p < pe_loop_times; p++)
				{
					//pe window shift
					//ap_uint < pe * DATA_WIDTH > tmp=0;	//store simd unit total multi-sum

					T_mac tmp[pe];
					for (int t = 0; t < pe; t++)
					{
					#pragma HLS UNROLL
						tmp[t] = 0;
					}
					loop4:
					for (int s = 0; s < simd_loop_times; s++)
					{
					#pragma HLS PIPELINE II = 1
						loop5:
							for (int k = 0; k < pe; k++)
							{
							#pragma HLS UNROLL
								if ((k > (left_pe - 1)) && (p == (pe_loop_times - 1)))
									break;

								tmp_i[k] = IFM[k].read();

								tmp_w[k] = WEIGHT[k].read();
								loop6:
									for (int n = 0; n < simd; n++)
									{
										//simd parallel
										#pragma HLS UNROLL
											in[n].range() = tmp_i[k].range((n + 1) *DATA_WIDTH - 1, n *DATA_WIDTH);
										wt[n].range() = tmp_w[k].range((n + 1) *WEIGHT_WIDTH - 1, n *WEIGHT_WIDTH);
										//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
										tmp_[n] = in[n] *wt[n];
										tmp[k] += tmp_[n];
										//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
									}
							}


					}

					for (int r = 0; r < pe; r++)
					{
						//traverse all tmp result
						#pragma HLS PIPELINE II = 1
						T_mac tmp_relu = max_op(tmp[r], (T_mac) 0);
						result = (half) tmp_relu;
						output.write(result);


						//write tmp to result one by one
						//debug2<<tmp[r].range(DATA_WIDTH-1,0)<<endl;


						//debug2<<"a"<<endl;
					}

					//debug2<<"b"<<endl;
				}

				//debug2<<"c"<<endl;
			}

			//debug2<<"d"<<endl;
		}

		//debug2<<"e"<<endl;
	}


//function -> padding matrix
//mul_result -> mul_result stream from Mac
//output -> output stream
//occupy_row_num -> each channel occupied row number. Eg. channel numer = 64, occupy_row_num = 2
//OFMDim -> OFMDim from previous layer. (After conv)
template <int DATA_WIDTH, int SIMD_I, int occupy_row_num, int OFMDim, int IN_T>
	void Padding(stream<ap_uint<DATA_WIDTH * SIMD_I>> &mul_result, stream<ap_uint<IN_T>> &output)
	{
		cout << "Padding" << endl;
		//debug2<<"constconstconstconstxxx"<<endl;
		loop1:
			for (int i = 0; i < OFMDim + 2; i++)
			{
				loop2: for (int j = 0; j < OFMDim + 2; j++)
				{
					loop3: for (int k = 0; k < occupy_row_num; k++)
					{
					#pragma HLS PIPELINE II = 1
						if (i == 0 || j == 0 || i == OFMDim + 1 || j == OFMDim + 1)
						{
							output.write(0);
							//debug2<<0<<" ";
						}
						else
						{
							ap_uint<DATA_WIDTH * SIMD_I> tmp = mul_result.read();
							output.write(tmp);
							//debug2<<tmp<<" ";
						}
					}
				}

				//debug2<<endl;
			}

		//debug2<<endl;
	}

//function -> max pooling matrix
//mul_result -> upper stream
//output -> output stream
//occupy_row_num -> each channel occupied row number. Eg. channel numer = 64, occupy_row_num = 2
//OFMDim -> OFMDim from previous layer. (After after padding)
template < typename T_data, int DATA_WIDTH, int SIMD_I, int occupy_row_num, int OFMDim >	//Question:for max pooling we need space to store the multiplication results from mac layer first
	void Pooling(stream<ap_uint<DATA_WIDTH * SIMD_I>> &mul_result, stream<ap_uint<DATA_WIDTH * SIMD_I>> &output)
	{
		cout << "Pooling" << endl;
		ap_uint<DATA_WIDTH * SIMD_I> tmp1[OFMDim / 2 *occupy_row_num];
		//#pragma HLS ARRAY_PARTITION variable=tmp1 complete dim=0
		ap_uint<DATA_WIDTH * SIMD_I> tmp2;

		ap_uint<DATA_WIDTH * SIMD_I> tmp4;

		//int ggg=0;

		loop1:
			for (ap_uint<16> i = 0; i < OFMDim / 2; i++)
			{
				loop2: for (ap_uint<16> j = 0; j < OFMDim / 2; j++)
				{
					loop3: for (ap_uint<16> k = 0; k < occupy_row_num; k++)
					{
					#pragma HLS PIPELINE II = 1
						tmp1[j *occupy_row_num + k] = mul_result.read();
					}

					loop4: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
					{
                        #pragma HLS PIPELINE II = 1 //todo: add this
						tmp2 = mul_result.read();
						ap_uint<DATA_WIDTH * SIMD_I> tmp3 = tmp1[j *occupy_row_num + l];
						T_data tmp5[SIMD_I];
						//#pragma HLS bind_storage variable=tmp5 type=RAM_1P impl=uram
						T_data tmp6[SIMD_I];
						//#pragma HLS bind_storage variable=tmp6 type=RAM_1P impl=uram
						for (ap_uint<16> q = 0; q < SIMD_I; q++)
						{
						#pragma HLS UNROLL
							tmp5[q].range() = tmp2((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH);
							tmp6[q].range() = tmp3((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH);
							if (tmp6[q] < tmp5[q])
							{
								tmp6[q] = tmp5[q];
							}

							tmp3((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH) = tmp6[q].range();
						}

						tmp1[j *occupy_row_num + l] = tmp3;
					}
				}

				loop6: for (ap_uint<16> j = 0; j < OFMDim / 2; j++)
				{
					loop7: for (ap_uint<16> k = 0; k < 2; k++)
					{
						loop8: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
						{
                            //#pragma HLS PIPELINE II = 1
							tmp4 = mul_result.read();
							ap_uint<DATA_WIDTH * SIMD_I> tmp3 = tmp1[j *occupy_row_num + l];
							ap_uint<DATA_WIDTH> tmp5[SIMD_I];
							ap_uint<DATA_WIDTH> tmp6[SIMD_I];
							loop9:
								for (ap_uint<16> m = 0; m < SIMD_I; m++)
								{
								#pragma HLS UNROLL
									tmp5[m].range() = tmp4((m + 1) *DATA_WIDTH - 1, m *DATA_WIDTH);
									tmp6[m].range() = tmp3((m + 1) *DATA_WIDTH - 1, m *DATA_WIDTH);
									if (tmp6[m] < tmp5[m])
									{
										tmp6[m] = tmp5[m];
									}

									tmp3((m + 1) *DATA_WIDTH - 1, m *DATA_WIDTH) = tmp6[m].range();
								}

							tmp1[j *occupy_row_num + l] = tmp3;
						}
					}
				}

				loop10: for (ap_uint<16> m = 0; m < OFMDim / 2; m++)
				{
					loop11: for (ap_uint<16> n = 0; n < occupy_row_num; n++)
					{
					#pragma HLS PIPELINE II = 1
						output.write(tmp1[m *occupy_row_num + n]);

						//debug3<<tmp1[m*occupy_row_num+n];
					}

					//debug3<<" ";
				}

				//debug3<<endl;
			}
	}

template <int DATA_WIDTH, int SIMD_I, int SIMD_I_2, int SIMD_I_NEXT, int occupy_row_num1, int OFMDim, int occupy_row_num2>
	void Concat(stream<ap_uint<DATA_WIDTH * SIMD_I>> &mul_result1, stream<ap_uint<DATA_WIDTH * SIMD_I_2>> &mul_result2, stream<ap_uint <DATA_WIDTH * SIMD_I_NEXT>> &Concat_output)
	{
		loop1: for (int i = 0; i < OFMDim; i++)
		{
			loop2: for (int j = 0; j < OFMDim; j++)
			{
				loop31: for (int k = 0; k < occupy_row_num1; k++)
				{
                    #pragma HLS PIPELINE II = 1 //todo: add this
					ap_uint<DATA_WIDTH * SIMD_I> tmp1;
					tmp1 = mul_result1.read();
					Concat_output.write(tmp1);
				}

				loop32: for (int k = 0; k < occupy_row_num2; k++)
				{
                    #pragma HLS PIPELINE II = 1  //todo: add this
					ap_uint<DATA_WIDTH * SIMD_I_2> tmp2;
					tmp2 = mul_result2.read();
					Concat_output.write(tmp2);
				}
			}
		}
	}

template <int DATA_WIDTH, int SIMD_I, int occupy_row_num, int OFMDim, int u_factor>
	void Upsample(stream<ap_uint<DATA_WIDTH * SIMD_I>> &mul_result, stream<ap_uint<DATA_WIDTH * SIMD_I>> &up_output)
	{
		ap_uint<DATA_WIDTH * SIMD_I> tmp1[OFMDim *occupy_row_num];
		//#pragma HLS ARRAY_PARTITION variable=tmp1 complete dim=0

		//int ggg=0;

		loop1:
			for (ap_uint<16> i = 0; i < u_factor * OFMDim; i++)
			{
				if (i % u_factor == 0)
				{
					loop2: for (ap_uint<16> j = 0; j < OFMDim; j++)
					{
						loop3: for (ap_uint<16> k = 0; k < occupy_row_num; k++)
						{
						#pragma HLS PIPELINE II = 1  //todo: add this
							tmp1[j *occupy_row_num + k] = mul_result.read();
						}
					}
				}

				loop4:
					for (ap_uint<16> m = 0; m < OFMDim; m++)
					{
						loop5: for (ap_uint<16> p = 0; p < u_factor; p++)
						{

							for (ap_uint<16> n = 0; n < occupy_row_num; n++)
							{
                               #pragma HLS PIPELINE II = 1
							 					//hls::print("value is: %d 
//", (int) tmp1[m*occupy_row_num+n]);
								up_output.write(tmp1[m *occupy_row_num + n]);
								//hls::print(" read value is: %d 
//", (int) up_output.read());
							}

							//debug3<<tmp1[m*occupy_row_num+n];
						}

						//debug3<<" ";
					}

				//debug3<<endl;
			}
	}



template <int DATA_WIDTH, int SIMD_I, typename T_data>
ap_uint<DATA_WIDTH * SIMD_I> compare(ap_uint<DATA_WIDTH * SIMD_I> a, ap_uint<DATA_WIDTH * SIMD_I> b) {
	T_data tmp5[SIMD_I];
	//#pragma HLS bind_storage variable=tmp5 type=RAM_1P impl=uram
	T_data tmp6[SIMD_I];
	//#pragma HLS bind_storage variable=tmp6 type=RAM_1P impl=uram
	for (ap_uint<16> q = 0; q < SIMD_I; q++)
	{
	#pragma HLS UNROLL
		tmp5[q].range() = a((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH);
		tmp6[q].range() = b((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH);
		if (tmp6[q] < tmp5[q])
		{
			tmp6[q] = tmp5[q];
		}

		b((q + 1) *DATA_WIDTH - 1, q *DATA_WIDTH) = tmp6[q].range();
	}

	return b;
}
// efficient stride pooling
template <typename T_data, int DATA_WIDTH, int SIMD_I, int occupy_row_num, int OFMDim>
void Pooling_stride_one(stream<ap_uint<DATA_WIDTH * SIMD_I>> &mul_result, stream<ap_uint<DATA_WIDTH * SIMD_I>> &output)
{
		cout << "Pooling" << endl;
		ap_uint<DATA_WIDTH * SIMD_I> tmp1[OFMDim *occupy_row_num];
		//#pragma HLS ARRAY_PARTITION variable=tmp1 complete dim=0

		ap_uint<DATA_WIDTH * SIMD_I> tmp1_nxt[2 *occupy_row_num];
        #pragma HLS ARRAY_PARTITION variable=tmp1_nxt complete dim=0


		ap_uint<DATA_WIDTH * SIMD_I> tmp1_nxt_cmp[occupy_row_num];

		ap_uint<DATA_WIDTH * SIMD_I> tmp2;

		ap_uint<DATA_WIDTH * SIMD_I> tmp4;

		//int ggg=0;


		loop10:
			for (ap_uint<16> j = 0; j < OFMDim; j++) // columns
			{
				loop03: for (ap_uint<16> k = 0; k < occupy_row_num; k++)
				{
				#pragma HLS PIPELINE II = 1
				tmp1[j *occupy_row_num + k] = mul_result.read();

				}

			}

		loop11:
			for (ap_uint<16> i = 1; i < OFMDim; i++) // columns
			{
				loop21: for (ap_uint<16> j = 0; j < OFMDim; j++) //rows
				{
					loop3: // for the previous rows
					for (ap_uint<16> l = 0; l < occupy_row_num; l++)
					{
						#pragma HLS PIPELINE II = 1 
						tmp1[(j-1) *occupy_row_num + l] = compare<DATA_WIDTH, SIMD_I, T_data>(tmp1[(j-1) *occupy_row_num + l], tmp1[j *occupy_row_num + l]);
					}

				}

				loop22: // for the last row
				for (ap_uint<16> l = 0; l < occupy_row_num; l++)
				{
					#pragma HLS PIPELINE II = 1 
					tmp1[(OFMDim-1) *occupy_row_num + l] = compare<DATA_WIDTH, SIMD_I, T_data>(tmp1[(OFMDim-1) *occupy_row_num + l], 0);
				}

				// read the next column (first element)
				loop23: for (ap_uint<16> k = 0; k < occupy_row_num; k++)
				{
					#pragma HLS PIPELINE II = 1
						tmp1_nxt[0 *occupy_row_num + k] = mul_result.read();
				}

				// read the following columns
				loop24: for (ap_uint<16> j = 1; j < OFMDim; j++)
				{
					loop241: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
					{
						#pragma HLS PIPELINE II = 1
						tmp1_nxt[1 *occupy_row_num + l] = mul_result.read();
						tmp1_nxt_cmp[l] = compare<DATA_WIDTH, SIMD_I, T_data>(tmp1_nxt[0 *occupy_row_num + l], tmp1_nxt[1 *occupy_row_num + l]);
						output.write(compare<DATA_WIDTH, SIMD_I, T_data>(tmp1_nxt_cmp[l], tmp1[(j-1) *occupy_row_num + l]));
						tmp1[(j-1) *occupy_row_num + l] = tmp1_nxt[0 *occupy_row_num + l];
						tmp1_nxt[0 *occupy_row_num + l] = tmp1_nxt[1 *occupy_row_num + l];
					}
				}

				loop34: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
				{
					#pragma HLS PIPELINE II = 1
					tmp1_nxt_cmp[l] = compare<DATA_WIDTH, SIMD_I, T_data>(tmp1_nxt[0 *occupy_row_num + l], 0);
					output.write(compare<DATA_WIDTH, SIMD_I, T_data>(tmp1_nxt_cmp[l], tmp1[(OFMDim-1) *occupy_row_num + l]));
					tmp1[(OFMDim-1) *occupy_row_num + l] = tmp1_nxt[0 *occupy_row_num + l];
				}

			}

		// compare with the last padded column
		loop12: for (ap_uint<16> j = 0; j < OFMDim-1; j++)
		{
			loop121: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
			{
				#pragma HLS PIPELINE II = 1
				output.write(compare<DATA_WIDTH, SIMD_I, T_data>(tmp1[j *occupy_row_num + l], tmp1[(j+1) *occupy_row_num + l]));
			}
		}
		loop13: for (ap_uint<16> l = 0; l < occupy_row_num; l++)
		{
			#pragma HLS PIPELINE II = 1
			output.write(compare<DATA_WIDTH, SIMD_I, T_data>(tmp1[(OFMDim-1) *occupy_row_num + l],0));
		}

}
            
#endif