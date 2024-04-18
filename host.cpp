//***************from finn*************************//
#include <iostream>
#include <time.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#include <ap_int.h>
//#include "bnn-library.h"
////#include "weights.hpp"
//#include "activations.hpp"
//#include "interpret.hpp"
//#include "mvau.hpp"
//#include "conv.hpp"

//***************from hybrid_PU*************************//
#include <vector>
#include <ap_fixed.h>
#include <CL/cl.h>
//#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include "xcl2.hpp"

//***************other libs*************************//
#include <fstream>
#include <math.h>
#include "define_.h"

using namespace hls;
using namespace std;

//*****************************following part are class or function definition********************************//
ofstream debug("./debug_file1.txt");
#define DATA_RANGE 9
enum data_type{FM,WEIGHT};

template<class T>
class input_data{
public:
	input_data(){};
	input_data(int k,int c,int d, data_type t, int s=3, int p=10);//construct function
	void matrix_gen();//based on input_data parameters allocate corresponding memory space for it, and then randomly assign value to it
	void matrix_<padding();//based on matrix that we create, allocate a new memory space and implement padding
	void organize_data();//organize IFM and weight into desired format
	void print_matrix();//print out the matrix
	void duplicate(int width);//for filters w>e duplicate them in the software side
	void print_vector();//print out the vector
	void soft_matrix_gen(int a, int b, int c);
	void pooling();
	input_data<T> &operator = (const input_data<T> &rhs);	//overload "=" operator to copy construct
	input_data operator * (input_data<T> & data2); //overload "*" operator to realize matrix conv
	vector<ap_uint<512>, aligned_allocator<ap_uint<512>> > return_vector();//used to avoid errors when passing data to hw side by opencl
private:
	int kernel,channel,dim;
	data_type dt;
	int simd;
	int pe;
	T ****matrix;
	vector<ap_uint<512>, aligned_allocator<ap_uint<512> > > data_vector;
	///////////////////////////////
	T ****soft_result;
	int result_kernel=1;
	int result_channel;
	int result_dim;
	data_type result_dt=FM;
};


template<class T>
input_data<T>::input_data(int k, int c, int d, data_type t,int s, int p){
	kernel=k;
	channel=c;
	dim=d;
	dt=t;
	simd=s;
	pe=p;
}

template<class T>
void input_data<T>::matrix_gen(){
	matrix=new T ***[kernel];
	for(int i=0;i<kernel;i++){
		matrix[i]=new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			matrix[i][j]=new T *[dim];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim;k++){
				matrix[i][j][k]=new T [dim];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					matrix[i][j][k][l]= rand()%(DATA_RANGE-1)+1;
				}
			}
		}
	}
}

template<class T>
void input_data<T>::matrix_padding(){
	T ****tmp = new T ***[kernel];
	for(int i=0; i< kernel; i++){
		tmp[i] = new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			tmp[i][j]=new T *[dim+2];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				tmp[i][j][k]=new T [dim+2];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				for(int l=0;l<dim+2;l++){
					tmp[i][j][k][l]= 0;
				}
			}
		}
	}


	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=1;k<dim+1;k++){
				for(int l=1;l<dim+1;l++){
					tmp[i][j][k][l]= matrix[i][j][k-1][l-1];
				}
			}
		}
	}

	//delete[] matrix;
	matrix = new T ***[kernel];
	for(int i=0; i< kernel; i++){
		matrix[i] = new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			matrix[i][j]=new T *[dim+2];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				matrix[i][j][k]=new T [dim+2];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				for(int l=0;l<dim+2;l++){
					matrix[i][j][k][l]= tmp[i][j][k][l];
				}
			}
		}
	}
	dim = dim + 2;
	//delete[] tmp;

}

template<class T>
void input_data<T>::organize_data(){
	if(dt==FM){
		for(int m=0;m<kernel;m++){
			for(int i=0;i<dim;i++){
				for(int j=0;j<dim;j++){
					ap_uint<512> tmp=0;
					for(int k=0;k<channel;k++){
						tmp((k+1)*ACTIVATION_PRECISION-1,k*ACTIVATION_PRECISION)=matrix[m][k][j][i];
					}
					data_vector.push_back(tmp);
				}
			}
		}
	}
	else{
		for(int i=0;i<kernel;i=i+pe){
			for(int l=0;l<channel;l=l+simd){
				for(int j=0;j<dim;j++){
					for(int k=0;k<dim;k++){
						for(int pp=0;pp<pe;pp++){
							if(simd>(512/WEIGHT_PRECISION)){
								ap_uint<512> tmp=0;
								for(int ss=0;ss<simd;ss=ss+(512/WEIGHT_PRECISION)){
									for(int n=0;n<512/WEIGHT_PRECISION;n++){
										tmp((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION)=matrix[i+pp][l+ss+n][k][j];

										if(n== (512/WEIGHT_PRECISION-1)){
											data_vector.push_back(tmp);
											tmp=0;
										}
									}
								}
							}
							else{
								ap_uint<512> tmp=0;
								for(int ss=0;ss<simd;ss++){
									tmp((ss+1)*WEIGHT_PRECISION-1,ss*WEIGHT_PRECISION)=matrix[i+pp][l+ss][k][j];
									//cout<<"/////"<<matrix[i+pp][l+ss][k][j]<<endl;
								}
								//cout<<tmp<<endl;
								//cout<<endl;

								data_vector.push_back(tmp);

							}
							if(kernel==(pp+i+1))
								break;
						}
					}
				}
			}
		}
	}
}

template<class T>
void input_data<T>::print_matrix(){
	for(int i=0;i<kernel;i++){
		if(dt==FM)
			debug<<"IFM:"<<endl;
		else if(dt==WEIGHT)
			debug<<"kernel #"<<i<<":"<<endl;
		for(int j=0;j<channel;j++){
			debug<<"channel #"<<j<<":"<<endl;
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					debug<<matrix[i][j][k][l]<<" ";
				}
				debug<<endl;
			}
			debug<<endl;
		}
		debug<<endl;
	}
}

template<class T>
void input_data<T>::duplicate(int width){
	int index=data_vector.size();
	for(int i=0;i<width*width-1;i++){
		for(int j=0;j<index;j++){
			data_vector.push_back(data_vector[j]);
		}
	}
}

template<class T>
void input_data<T>::print_vector(){
	for(unsigned int i=0;i<data_vector.size();i++)
		debug<<data_vector[i]<<endl;
}

template<class T>
void input_data<T>::pooling(){
	T ****tmp=new T ***[kernel];
		for(int i=0;i<kernel;i++){
			tmp[i]=new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				tmp[i][j]=new T *[dim/2];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					tmp[i][j][k]=new T [dim/2];
				}
			}
		}
		//processing part
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					for(int l=0;l<dim/2;l++){
						int tmp_max=-99999;
						for(int m=0;m<2;m++){
							for(int n=0;n<2;n++){
								if(matrix[i][j][2*k+m][2*l+n]>tmp_max)
									tmp_max=matrix[i][j][2*k+m][2*l+n];
							}
						}
						tmp[i][j][k][l]= tmp_max;
					}
				}
			}
		}

		delete[] matrix;
		T ****matrix = new T ***[kernel];
		for(int i=0; i< kernel; i++){
			matrix[i] = new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				matrix[i][j]=new T *[dim/2];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					matrix[i][j][k]=new T [dim/2];
				}
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					for(int l=0;l<dim/2;l++){
						matrix[i][j][k][l]= tmp[i][j][k][l];
					}
				}
			}
		}
		dim=dim/2;

}

template<class T>
input_data<T> &input_data<T>::operator = (const input_data<T> &rhs){

		kernel=rhs.result_kernel;
		channel=rhs.result_channel;
		dim=rhs.result_dim;
		dt=rhs.result_dt;

		matrix=new T ***[kernel];
		for(int i=0;i<kernel;i++){
			matrix[i]=new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				matrix[i][j]=new T *[dim];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim;k++){
					matrix[i][j][k]=new T [dim];
				}
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim;k++){
					for(int l=0;l<dim;l++){
						matrix[i][j][k][l]= rhs.soft_result[i][j][k][l];
					}
				}
			}
		}

	return *this;
}

template<class T>
input_data<T> input_data<T>::operator * (input_data<T> &data2){

	result_channel=data2.kernel;
	result_dim=dim-data2.dim+1;

	soft_result=new T ***[result_kernel];
	for(int i=0;i<result_kernel;i++){
		soft_result[i]=new T **[result_channel];
	}
	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			soft_result[i][j]=new T *[result_dim];
		}
	}
	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			for(int k=0;k<result_dim;k++){
				soft_result[i][j][k]=new T [result_dim];
			}
		}
	}

	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			for(int k=0;k<result_dim;k++){
				for(int l=0;l<result_dim;l++){
					soft_result[i][j][k][l]=0;
				}
			}
		}
	}

	for(int i=0;i<result_kernel;i++){

			for(int k=0;k<result_dim;k++){
				for(int l=0;l<result_dim;l++){

					for(int m=0;m<data2.kernel;m++){
						for(int n=0;n<data2.channel;n++){
							for(int p=0;p<data2.dim;p++){
								for(int q=0;q<data2.dim;q++){
									soft_result[i][m][k][l]+=matrix[i][n][k+p][l+q]*data2.matrix[m][n][p][q];
								}
							}
						}
					}
				}

		}
	}

	return *this;
}




template<class T>
vector<ap_uint<512>, aligned_allocator<ap_uint<512>> > input_data<T>::return_vector(){
	return data_vector;
}

int main(int argc, char** argv){
	if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
	//generate all needed data
	srand((int) time(0));
	input_data<ap_uint<ACTIVATION_PRECISION> > IFM(1,IFM_Channels_1,IFMDim_1,FM);
	input_data<ap_uint<WEIGHT_PRECISION> > W1(OFM_Channels_1,IFM_Channels_1,KERNELDim_1,WEIGHT,SIMD_1,PE_1);
	input_data<ap_uint<WEIGHT_PRECISION> > W2(OFM_Channels_2,IFM_Channels_2,KERNELDim_2,WEIGHT,SIMD_2,PE_2);
	input_data<ap_uint<WEIGHT_PRECISION> > W3(OFM_Channels_3,IFM_Channels_3,KERNELDim_3,WEIGHT,SIMD_3,PE_3);
	input_data<ap_uint<WEIGHT_PRECISION> > W4(OFM_Channels_4,IFM_Channels_4,KERNELDim_4,WEIGHT,SIMD_4,PE_4);
	input_data<ap_uint<WEIGHT_PRECISION> > W5(OFM_Channels_5,IFM_Channels_5,KERNELDim_5,WEIGHT,SIMD_5,PE_5);
	input_data<ap_uint<WEIGHT_PRECISION> > W6(OFM_Channels_6,IFM_Channels_6,KERNELDim_6,WEIGHT,SIMD_6,PE_6);
	input_data<ap_uint<WEIGHT_PRECISION> > W7(OFM_Channels_7,IFM_Channels_7,KERNELDim_7,WEIGHT,SIMD_7,PE_7);
	input_data<ap_uint<WEIGHT_PRECISION> > W8(OFM_Channels_8,IFM_Channels_8,KERNELDim_8,WEIGHT,SIMD_8,PE_8);
	input_data<ap_uint<WEIGHT_PRECISION> > W9(OFM_Channels_9,IFM_Channels_9,KERNELDim_9,WEIGHT,SIMD_9,PE_9);

	input_data<ap_uint<ACTIVATION_PRECISION> > OFM1;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM2;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM3;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM4;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM5;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM6;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM7;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM8;
	input_data<ap_uint<ACTIVATION_PRECISION> > OFM9;

	IFM.matrix_gen();
	W1.matrix_gen();
	W2.matrix_gen();
	W3.matrix_gen();
	W4.matrix_gen();
	W5.matrix_gen();
	W6.matrix_gen();
	W7.matrix_gen();
	W8.matrix_gen();
	W9.matrix_gen();

	W1.organize_data();
	W2.organize_data();
	W3.organize_data();
	W4.organize_data();
	W5.organize_data();
	W6.organize_data();
	W7.organize_data();
	W8.organize_data();
	W9.organize_data();

	IFM.matrix_padding();
	IFM.organize_data();
	OFM1 = IFM * W1;
	OFM1.pooling();
	OFM1.matrix_padding();
	OFM2 = OFM1 * W2;
	OFM2.pooling();
	OFM2.matrix_padding();
	OFM3 = OFM2 * W3;
	OFM3.pooling();
	OFM3.matrix_padding();
	OFM4 = OFM3 * W4;
	OFM4.pooling();
	OFM4.matrix_padding();
	OFM5 = OFM4 * W5;
	OFM5.matrix_padding();
	OFM6 = OFM5 * W6;
	OFM7 = OFM6 * W7;
	OFM7.matrix_padding();
	OFM8 = OFM7 * W8;
	OFM9 = OFM8 * W9;

	cout<<"Here0"<<endl;
	int out_size = 1*1*1*WIDTH/8;
    vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>>> output_vector(out_size);
    vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > IFM_(IFM.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W1_(W1.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W2_(W2.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W3_(W3.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W4_(W4.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W5_(W5.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W6_(W6.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W7_(W7.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W8_(W8.return_vector());
	vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W9_(W9.return_vector());

	int IFM_size = IFM.return_vector().size()*WIDTH/8;
	int weight1_size=W1_.size()*WIDTH/8;//byte
	int weight2_size=W2_.size()*WIDTH/8;//byte
	int weight3_size=W3_.size()*WIDTH/8;//byte
	int weight4_size=W4_.size()*WIDTH/8;//byte
	int weight5_size=W5_.size()*WIDTH/8;//byte
	int weight6_size=W6_.size()*WIDTH/8;//byte
	int weight7_size=W7_.size()*WIDTH/8;//byte
	int weight8_size=W8_.size()*WIDTH/8;//byte
	int weight9_size=W9_.size()*WIDTH/8;//byte

	//Open_cl host code area start
    cl_int err;
	std::string binaryFile = argv[1];
	auto devices = xcl::get_xil_devices();
	auto device = devices[0];

	OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(),fileBuf.size()}};
	devices.resize(1);
	OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
	OCL_CHECK(err, cl::Kernel one_layer(program, "ultranet_latest", &err));
	// calculate buffer size

	//allocate buffer in global memory
	cout<<"***host allocate buffer in global memory***"<<endl;

	OCL_CHECK(err, cl::Buffer buffer_IFM(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, IFM_size, IFM_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight1_size, W1_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight2(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight2_size, W2_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight3(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight3_size, W3_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight4(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight4_size, W4_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight5(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight5_size, W5_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight6(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight6_size, W6_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight7(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight7_size, W7_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight8(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight8_size, W8_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight9(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight9_size, W9_.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, out_size, output_vector.data(), &err));

	//set the Kernel Arguments
	cout<<"***host set the Kernel Arguments***"<<endl;
	int narg=0;
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_IFM));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight1));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight2));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight3));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight4));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight5));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight6));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight7));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight8));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight9));
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_out));
//copy data from host to device
	cout<<"***host copy data from host to device***"<<endl;
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_IFM}, 0 /* 0 means from host*/));
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_weight1,
													 buffer_weight2,
													 buffer_weight3,
													 buffer_weight4,
													 buffer_weight5,
													 buffer_weight6,
													 buffer_weight7,
													 buffer_weight8,
													 buffer_weight9}, 0 /* 0 means from host*/));
	// Launch the Kernel
	cout<<"***host Launch the Kernel***"<<endl;
    OCL_CHECK(err, err = q.enqueueTask(one_layer));

	// Copy Result from Device Global Memory to Host Local Memory
	cout<<"***host Copy Result from Device Global Memory to Host Local Memory***"<<endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    cout<<"***here***"<<endl;
    q.finish();
    cout<<"***end***"<<endl;
	return 0;
}