
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <direct.h>
#include "newsketch.cuh"
#include "fstream"
#include "GeneralUtil.cuh"
//#include "sm_35_atomic_functions.h"
#include<vector>
typedef unsigned long long uint64;
typedef unsigned int uint32;
typedef unsigned char uint8;

using namespace std;
GeneralSketchBloom *GSB;
GeneralSketchBloom *GSB1;
GeneralVSketch *GVS;
GeneralVSketch *GVS1;
unsigned int *packetdata;
__global__ void initsketch(GeneralSketchBloom *GSB,GeneralVSketch *GVS, int GSB_or_GVS,int size_or_spread,int number,unsigned int* packetdata) {
	int id = blockIdx.x *blockDim.x + threadIdx.x;
	if (id >= number) return;
	uint32_t srcIP =  packetdata[id * 2];
	uint32_t dstIP =  packetdata[id * 2 + 1];
	uint32_t x;
	uint32_t j;
	if (GSB_or_GVS == 0) {
		for (int i = 0; i < 1; i++) {
			x = uIntHash(srcIP);
			if (size_or_spread == 0) {
				if (*GSB->sketchName == 0) {
					for (int pp = 0; pp < 4; pp++) {
						j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
						encodeCounter(GSB->C[0][j]);
					}
				}
				else
					if (*GSB->sketchName == 1) {

						for (int pp = 0; pp < 4; pp++) {
							j = (intHash(x ^ GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
							encodeBitmap(GSB->B[0][j]);
						}
					}
					else
						if (*GSB->sketchName == 2) {
							for (int pp = 0; pp < 4; pp++) {
								j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
								encodeFMsketch(GSB->F[0][j]);
							}
						}
						else
							if (*GSB->sketchName == 3) {
								for (int pp = 0; pp < 4; pp++) {
									j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
									encodeHyperLogLog(GSB->H[0][j]);
								}
							}
			}
			else {
				if (*GSB->sketchName == 0) {
					for (int pp = 0; pp < 4; pp++) {
						j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
						encodeCounterEID(GSB->C[0][j], srcIP);
					}
				}
				else
					if (*GSB->sketchName == 1) {
						for (int pp = 0; pp < 4; pp++) {
							j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
							encodeBitmapEID(GSB->B[0][j], srcIP);
						}
					}
					else
						if (*GSB->sketchName == 2) {
							for (int pp = 0; pp < 4; pp++) {
								j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
								encodeFMsketchEID(GSB->F[0][j], srcIP);
							}
						}
						else
							if (*GSB->sketchName == 3) {
								for (int pp = 0; pp < 4; pp++) {
									j = (intHash(x^GSB->S[pp]) % *GSB->w + *GSB->w) % *GSB->w;
									encodeHyperLogLogEID(GSB->H[0][j], srcIP);
								}
							}
			}
		}
	}
	else if(GSB_or_GVS==1){
		int w_m = *GVS->w / *GVS->m;
		for (int i = 0; i < 1; i++) {
			if (size_or_spread == 0) {
				if (*GVS->sketchName == 0)
					encodeCounterSegment(GVS->C[0], srcIP, GVS->S, w_m);
				else
					if (*GVS->sketchName == 1)
						encodeBitmapSegment(GVS->B[0], srcIP, GVS->S, w_m);
					else
						if (*GVS->sketchName == 2)
							encodeFMsketchSegment(GVS->F[0], srcIP, GVS->S, w_m);
						else
							if (*GVS->sketchName == 3)
								encodeHyperLogLogSegment(GVS->H[0], srcIP, GVS->S, w_m);
			}
			else {
				if (*GVS->sketchName == 0)
					encodeCounterSegmentEID(GVS->C[0], srcIP, dstIP, GVS->S, w_m);
				else
					if (*GVS->sketchName == 1)
						encodeBitmapSegmentEID(GVS->B[0], srcIP, dstIP, GVS->S, w_m);
					else
						if (*GVS->sketchName == 2)
							encodeFMsketchSegmentEID(GVS->F[0], srcIP, dstIP, GVS->S, w_m);
						else
							if (*GVS->sketchName == 3)
								encodeHyperLogLogSegmentEID(GVS->H[0], srcIP, dstIP, GVS->S, w_m);
			}
		}
	}
}

void getoutputGSB(GeneralSketchBloom *GSB,int sketch_name,int size_or_spread) {
	int w;
	GeneralSketchBloom tmp1;
	cudaMemcpy(&tmp1, GSB, sizeof(GeneralSketchBloom), cudaMemcpyDeviceToHost);
	cudaMemcpy(&w, tmp1.w, sizeof(int), cudaMemcpyDeviceToHost);
	if (sketch_name == 0) {
		Counter ***x = (Counter ***)malloc(sizeof(Counter **) * 1);
		cudaMemcpy(x, tmp1.C, sizeof(Counter **), cudaMemcpyDeviceToHost);
		Counter **tmpC = (Counter **)malloc(w * sizeof(Counter *));
		cudaMemcpy(tmpC, x[0], sizeof(Counter *)* w, cudaMemcpyDeviceToHost);
		int GSBS[4];
		cudaMemcpy(GSBS, tmp1.S, sizeof(int) * 4, cudaMemcpyDeviceToHost);
		ofstream out;
		if (size_or_spread == 0)
			out.open("..\\..\\result\\BSketch\\gpu_counter_size_out.txt", ios::out);
		else
			out.open("..\\..\\result\\BSketch\\gpu_counter_spread_out.txt", ios::out);
		for (int i = 0; i < w;i++) {
			Counter p;
			cudaMemcpy(&p, tmpC[i], sizeof(Counter), cudaMemcpyDeviceToHost);
			int m;
			cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
			//cout << m << endl;
			int *o=(int *)malloc(sizeof(int)*m);
			cudaMemcpy(o, p.counters, m * sizeof(int), cudaMemcpyDeviceToHost);
			for (int j = 0; j < m; j++) {
				out << o[j];
				out << endl;
			}
		}
		out.close();
	}
	else 
		if (sketch_name == 1) {
			Bitmap ***x = (Bitmap ***)malloc(sizeof(Bitmap **) * 1);
			cudaMemcpy(x, tmp1.B, sizeof(Bitmap **), cudaMemcpyDeviceToHost);
			Bitmap **tmpB = (Bitmap **)malloc(w * sizeof(Bitmap *));
			cudaMemcpy(tmpB, x[0], sizeof(Bitmap *)* w, cudaMemcpyDeviceToHost);
			int GSBS[4];
			cudaMemcpy(GSBS, tmp1.S, sizeof(int) * 4, cudaMemcpyDeviceToHost);
			ofstream out;
			if (size_or_spread == 0)
				out.open("..\\..\\result\\BSketch\\gpu_bitmap_size_out.txt", ios::out);
			else
				out.open("..\\..\\result\\BSketch\\gpu_bitmap_spread_out.txt", ios::out);
			for (int i = 0; i < w;i++) {
				Bitmap p;
				cudaMemcpy(&p, tmpB[i], sizeof(Bitmap), cudaMemcpyDeviceToHost);
				int m;
				cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
				//cout << m << endl;
				bool *o=(bool *)malloc(sizeof(bool)*m);
				cudaMemcpy(o, p.B, m * sizeof(bool), cudaMemcpyDeviceToHost);
				for (int j = 0; j < m; j++) {
					out << o[j];
					out << endl;
				}
			}
			out.close();
		}
		else
			if (sketch_name == 2) {
				FMsketch ***x = (FMsketch ***)malloc(sizeof(FMsketch **) * 1);
				cudaMemcpy(x, tmp1.F, sizeof(FMsketch **), cudaMemcpyDeviceToHost);
				FMsketch **tmpF = (FMsketch **)malloc(w * sizeof(FMsketch *));
				cudaMemcpy(tmpF, x[0], sizeof(FMsketch *)* w, cudaMemcpyDeviceToHost);
				int GSBS[4];
				cudaMemcpy(GSBS, tmp1.S, sizeof(int) * 4, cudaMemcpyDeviceToHost);
				ofstream out;
				if (size_or_spread == 0)
					out.open("..\\..\\result\\BSketch\\gpu_fm_size_out.txt", ios::out);
				else
					out.open("..\\..\\result\\BSketch\\gpu_fm_spread_out.txt", ios::out);
				for (int i = 0; i < w;i++) {
					FMsketch p;
					cudaMemcpy(&p, tmpF[i], sizeof(FMsketch), cudaMemcpyDeviceToHost);
					int m;
					cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
					int size;			
					cudaMemcpy(&size, p.FMsketchSize, sizeof(int), cudaMemcpyDeviceToHost);
					//cout << m << " " << size << endl;
					bool **o=(bool **)malloc(sizeof(bool *)*m);
					cudaMemcpy(o, p.FMsketchMatrix, m * sizeof(bool*), cudaMemcpyDeviceToHost);
					for (int j = 0; j < m; j++) {
						bool *r= (bool *)malloc(sizeof(bool)*size);
						cudaMemcpy(r, o[j], size * sizeof(bool), cudaMemcpyDeviceToHost);
						for (int l = 0; l < size; l++) {
							out << r[l];
							out << endl;
						}
					}
				}
				out.close();
			}
			else
				if (sketch_name == 3) {
					HyperLogLog ***x = (HyperLogLog ***)malloc(sizeof(HyperLogLog **) * 1);
					cudaMemcpy(x, tmp1.H, sizeof(HyperLogLog **), cudaMemcpyDeviceToHost);
					HyperLogLog **tmpH = (HyperLogLog **)malloc(w * sizeof(HyperLogLog *));
					cudaMemcpy(tmpH, x[0], sizeof(HyperLogLog *)* w, cudaMemcpyDeviceToHost);
					int GSBS[4];
					cudaMemcpy(GSBS, tmp1.S, sizeof(int) * 4, cudaMemcpyDeviceToHost);
					ofstream out;
					if (size_or_spread == 0)
						out.open("..\\..\\result\\BSketch\\gpu_hll_size_out.txt", ios::out);
					else
						out.open("..\\..\\result\\BSketch\\gpu_hll_spread_out.txt", ios::out);
					for (int i = 0; i < w; i++) {
						HyperLogLog p;
						cudaMemcpy(&p, tmpH[i], sizeof(HyperLogLog), cudaMemcpyDeviceToHost);
						int m;
						cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
						int size;
						cudaMemcpy(&size, p.HLLSize, sizeof(int), cudaMemcpyDeviceToHost);
						//cout << m << " " << size << endl;
						bool **o = (bool **)malloc(sizeof(bool *)*m);
						cudaMemcpy(o, p.HLL, m * sizeof(bool*), cudaMemcpyDeviceToHost);
						for (int j = 0; j < m; j++) {
							bool *r = (bool *)malloc(sizeof(bool)*size);
							cudaMemcpy(r, o[j], size * sizeof(bool), cudaMemcpyDeviceToHost);
							for (int l = 0; l < size; l++) {
								out << r[l];
								out << endl;
							}
						}
					}
					out.close();
				}
}
void getoutputGVS(GeneralVSketch *GVS,int sketch_name,int size_or_spread) {
	int w;
	GeneralVSketch tmp1;
	cudaMemcpy(&tmp1, GVS, sizeof(GeneralVSketch), cudaMemcpyDeviceToHost);
	cudaMemcpy(&w, tmp1.w, sizeof(int), cudaMemcpyDeviceToHost);
	//cout << w << endl;
	if (sketch_name == 0) {
		Counter **x = (Counter **)malloc(sizeof(Counter *) * 1);
		cudaMemcpy(x, tmp1.C, sizeof(Counter *) * 1, cudaMemcpyDeviceToHost);
		ofstream out;
		if(size_or_spread==0)
			out.open("..\\..\\result\\VSketch\\gpu_counter_size_out.txt", ios::out);
		else
			out.open("..\\..\\result\\VSketch\\gpu_counter_spread_out.txt", ios::out);
		Counter p;
		cudaMemcpy(&p, x[0], sizeof(Counter), cudaMemcpyDeviceToHost);
		int m;
		cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
		//cout << m << endl;
		int *o = (int *)malloc(sizeof(int)*m);
		cudaMemcpy(o, p.counters, m * sizeof(int), cudaMemcpyDeviceToHost);
		for (int j = 0; j < m; j++) {
			out << o[j];
			out << endl;
		}
		out.close();
	}
	else
		if (sketch_name == 1) {
			Bitmap **x = (Bitmap **)malloc(sizeof(Bitmap *) * 1);
			cudaMemcpy(x, tmp1.B, sizeof(Bitmap *) * 1, cudaMemcpyDeviceToHost);
			ofstream out;
			if (size_or_spread == 0)
				out.open("..\\..\\result\\VSketch\\gpu_bitmap_size_out.txt", ios::out);
			else
				out.open("..\\..\\result\\VSketch\\gpu_bitmap_spread_out.txt", ios::out);
			Bitmap p;
			cudaMemcpy(&p, x[0], sizeof(Bitmap), cudaMemcpyDeviceToHost);
			int m;
			cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
			//cout << m << endl;
			bool *o = (bool *)malloc(sizeof(bool)*m);
			cudaMemcpy(o, p.B, m * sizeof(bool), cudaMemcpyDeviceToHost);
			for (int j = 0; j < m; j++) {
				out << o[j];
				out << endl;
			}
			out.close();
		}
		else
			if (sketch_name == 2) {
				FMsketch **x = (FMsketch **)malloc(sizeof(FMsketch *) * 1);
				cudaMemcpy(x, tmp1.F, sizeof(FMsketch *), cudaMemcpyDeviceToHost);
				ofstream out;
				if (size_or_spread == 0)
					out.open("..\\..\\result\\VSketch\\gpu_fm_size_out.txt", ios::out);
				else
					out.open("..\\..\\result\\VSketch\\gpu_fm_spread_out.txt", ios::out);
				FMsketch p;
				cudaMemcpy(&p, x[0], sizeof(FMsketch), cudaMemcpyDeviceToHost);
				int m;
				cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
				int size;
				cudaMemcpy(&size, p.FMsketchSize, sizeof(int), cudaMemcpyDeviceToHost);
				//cout << m << " " << size << endl;
				bool **o = (bool **)malloc(sizeof(bool *)*m);
				cudaMemcpy(o, p.FMsketchMatrix, m * sizeof(bool*), cudaMemcpyDeviceToHost);
				for (int j = 0; j < m; j++) {
					bool *r = (bool *)malloc(sizeof(bool)*size);
					cudaMemcpy(r, o[j], size * sizeof(bool), cudaMemcpyDeviceToHost);
					for (int l = 0; l < size; l++) {
						out << r[l];
						out << endl;
					}
				}
				out.close();
			}
			else
				if (sketch_name == 3) {
					HyperLogLog **x = (HyperLogLog **)malloc(sizeof(HyperLogLog *) * 1);
					cudaMemcpy(x, tmp1.H, sizeof(HyperLogLog **), cudaMemcpyDeviceToHost);
					ofstream out;
					if (size_or_spread == 0)
						out.open("..\\..\\result\\VSketch\\gpu_hll_size_out.txt", ios::out);
					else
						out.open("..\\..\\result\\VSketch\\gpu_hll_spread_out.txt", ios::out);
					HyperLogLog p;
					cudaMemcpy(&p, x[0], sizeof(HyperLogLog), cudaMemcpyDeviceToHost);
					int m;
					cudaMemcpy(&m, p.m, sizeof(int), cudaMemcpyDeviceToHost);
					int size;
					cudaMemcpy(&size, p.HLLSize, sizeof(int), cudaMemcpyDeviceToHost);
					//cout << m << " " << size << endl;
					bool **o = (bool **)malloc(sizeof(bool *)*m);
					cudaMemcpy(o, p.HLL, m * sizeof(bool*), cudaMemcpyDeviceToHost);
					for (int j = 0; j < m; j++) {
						bool *r = (bool *)malloc(sizeof(bool)*size);
						cudaMemcpy(r, o[j], size * sizeof(bool), cudaMemcpyDeviceToHost);
						for (int l = 0; l < size; l++) {
							out << r[l];
							out << endl;
						}
					}
					out.close();
				}
}

int readdata(string filename,char** data) {
	ifstream in;
	in.open(filename);
	
	char t[50];
	int number=0;
	while (in.getline(t, 40)) number++;
	in.close();
	char** tmpdata;
	tmpdata = (char **)malloc(sizeof(char *)*number);
	cudaMalloc((void **)&data,sizeof(char *)*number);
	in.open(filename);
	for (int i = 0; i < number;i++) {
		in.getline(t, 40);
		cudaMalloc((void **)&tmpdata[i], sizeof(char) * 40);
		cudaMemcpy(tmpdata[i],t,sizeof(char)*40,cudaMemcpyHostToDevice);
	}
	cudaMemcpy(data, tmpdata, sizeof(char *)*number, cudaMemcpyHostToDevice);
	in.close();
	return number;
}

void ip_str_to_num1(unsigned int *src, unsigned int *dst, char *buf) {
	sscanf(buf, "%u%u", src,dst);
}

unsigned int *readdata1(int *number,string filename,unsigned int* data) {
	ifstream in;
	in.open(filename);
	char t[50];
	*number = 0;
	while (in.getline(t, 40)) (*number)++;
	in.close();
	cout << *number << endl;
	size_t pitch=0;
	unsigned int *tmpdata;
	cudaMalloc((void **)&data, sizeof(unsigned int)*2* *number);
	tmpdata = (unsigned int *)malloc(sizeof(unsigned int) * 2 * *number);
	in.open(filename);
	unsigned int x, y;
	for (int i = 0; i < *number; i++) {
		in.getline(t, 40);
		ip_str_to_num1(&x, &y, t);
		tmpdata[i * 2] = x;
		tmpdata[i * 2 + 1] = y;
		if (i % 1000000 == 0) cout << i<<endl;
	}
	in.close();
	cudaMemcpy(data, tmpdata,sizeof(unsigned int) * *number, cudaMemcpyHostToDevice);
	free(tmpdata);
	return data;
}

void cudaFreepacket(char **data,int number) {
	char** tmpdata;
	tmpdata = (char **)malloc(sizeof(char *)*number);
	cudaMemcpy(tmpdata, data, sizeof(char *)*number, cudaMemcpyDeviceToHost);
	for (int i = 0; i < number; i++) {
		cudaFree(tmpdata[i]);
	}
	cudaFree(data);
	free(tmpdata);
}

void cudaFreepacket1(unsigned int *data) {
	cudaFree(data);
}

void experiment_start() {
	uint32 len;
	string filename = "..\\..\\data\\srcdstsize.txt";
	int number=10000000;
	packetdata=readdata1(&number,filename,packetdata);
	int *cudanumber;
	cudaMalloc((void**)&cudanumber, sizeof(int));
	cudaMemcpy(cudanumber, &number, sizeof(int), cudaMemcpyHostToDevice);
	initcurand << <1, 1 >> > ();
	cudaError_t cudaStatus = cudaThreadSynchronize();
	for (int lll = 0; lll < number_of_test; lll++) {
		if(GSB_or_GVS[lll]==0)
			GSB = initSketchBloom(sketch_name[lll]);
		else
			if(GSB_or_GVS[lll] == 1)
				GVS =initVSketch(sketch_name[lll]);
		cudaStatus = cudaThreadSynchronize();
		cudaEvent_t startEvent, stopEvent;
		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);
		cudaEventRecord(startEvent, 0);
		initsketch << <(number / 1024 + 1), 1024 >> > (GSB, GVS,  GSB_or_GVS[lll], size_or_spread[lll], number, packetdata);
		cudaEventRecord(stopEvent, 0);
		cudaStatus = cudaEventSynchronize(stopEvent);
		float time;
		cudaEventElapsedTime(&time, startEvent, stopEvent);
		cout << "GVS or GSB="<<GSB_or_GVS[lll]<<" size_or_spread="<<size_or_spread[lll]<<" sketchname="<<sketch_name[lll] <<" time=" << number / time / 1000 << "million packets/ms" << endl;
		cudaMemcpy(&number,cudanumber, sizeof(int), cudaMemcpyDeviceToHost);
		cout << "number=" << number << endl;
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);
		//if you want output, please use these code
		/*/if (GSB_or_GVS[lll] == 0)
			getoutputGSB(GSB, sketch_name[lll], size_or_spread[lll]);
		else
			getoutputGVS(GVS,sketch_name[lll], size_or_spread[lll]);/*/
		if (GSB_or_GVS[lll] == 0)cudaFree(GSB);
		else
			if (GSB_or_GVS[lll] == 1)
				cudaFree(GVS);
	}
}

int main() {
	experiment_start();
	return 0;
}
