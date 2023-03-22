/* GPU-based correlation code for a small (<=64) input
 * sampling system to run on a workstation class computer. There is an
 * upper limit to the number of supported channels due to the max number
 * of threads that can be on the GPU. This is currently 512.
 *
 * Author: Randall Wayth. Feb, 2009.
 *
 * to compile with CUDA: nvcc -O -o corr_gpu_complex corr_gpu_complex.cu -lcufft
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <ctype.h>
//#include "/usr/include/complex.h"  // nvcc stuffs up for some reason without the full path name
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <fcntl.h>

//#include "/home/rwayth/Desktop/gxc_kernels.h"

#define MAX_THREADS 128
#define MAX_INPUTS 256
#define MAX_CORR_PROD (MAX_INPUTS*(MAX_INPUTS+1)/2)
#define MAX_CHAN 128
//#define GRPSIZ 2    // this cannot be changed from 2 without rewriting the 1xG CMAC function.
#define INTMULT(a,b) __mul24((a),(b))
//#define INTMULT(a,b) (a)*(b)

#define TILE_SIZE_UNPACK 16 // note: if you change this, you must change LOG2_TILESIZE too!
#define LOG2_TILESIZE 4
#define UNPACK_BLOCK_ROWS (256/TILE_SIZE_UNPACK)

#define BITS_PER_NUM 5
#define TWOCOMP_MAXUINT   (1<<(BITS_PER_NUM))    // = 32 for 5 bits
#define TWOCOMP_ZEROPOINT (1<<(BITS_PER_NUM-1))  // = 16 for 5 bits
#define DECODER_MASK (TWOCOMP_MAXUINT -1)

#define BITS_PER_NUM_8BIT 4
#define TWOCOMP_MAXUINT_8BIT   (1<<(BITS_PER_NUM_8BIT))    // = 16 for 4 bits
#define TWOCOMP_ZEROPOINT_8BIT (1<<(BITS_PER_NUM_8BIT-1))  // = 8  for 4 bits
#define DECODER_MASK_8BIT (TWOCOMP_MAXUINT_8BIT -1)

enum {CMAC_1x1, CMAC_1xG_2, CMAC_1xG_4, CMAC_GxG_2, CMAC_GxG_4, CMAC_END};
enum {DATATYPE_MWA16, DATATYPE_MWA8, DATATYPE_COMPLEX_SHORT, DATATYPE_COMPLEX_FLOAT,
    DATATYPE_SIGNED_BYTE,DATATYPE_UNSIGNED_BYTE,DATATYPE_SIGNED_SHORT,DATATYPE_UNSIGNED_SHORT,
    DATATYPE_FLOAT,DATATYPE_COMPLEX_UNSIGNED_BYTE,DATATYPE_COMPLEX_SIGNED_BYTE,DATATYPE_REAL_UNSIGNED_4BIT};

/* function prototypes */
void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[], const char *optstring);
int readData(int nchan,int ninp,FILE *fpin,unsigned char *inp_buf);
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc);
int do_FFT_gpu(int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf);
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                 cuFloatComplex *buf,float normaliser);
void do_CMAC_gpu(int method, const int nchan, const int ninp, const int batchsize,
         cufftComplex * const ft_buf, cufftComplex * const corr_buf);
__global__ void do_CMAC_gpu_1xG(const int nchan, const int ninp, const int batchsize,
         cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void do_CMAC_gpu_1xG_4(const int nchan, const int ninp, const int batchsize,
         cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void do_CMAC_gpu_1x1(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void do_CMAC_gpu_GxG(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void do_CMAC_gpu_GxG_4(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void unpack_data_GPU_MWA16bit_transpose(const int nchan,const int ninp,const int batchsize, int log2_nchan,
                unsigned short *in_buf, cuFloatComplex *out_buf);
__global__ void unpack_data_GPU_MWA16bit_simple(const int nchan,const int ninp,const int batchsize, unsigned short *staging_buf,
                cuFloatComplex *inp_buf);
__global__ void unpack_data_GPU_16bit_simple(const int nchan,const int ninp,const int batchsize, char2 *staging_buf,
                cuFloatComplex *inp_buf);
__global__ void unpack_data_GPU_uint8_complex_simple(const int nchan,const int ninp,const int batchsize, char2 *staging_buf,
                cuFloatComplex *inp_buf);
__global__ void unpack_data_GPU_MWA8bit_simple(const int nchan,const int ninp,const int batchsize, unsigned char *staging_buf,
                cuFloatComplex *inp_buf_gpu);
__global__ void unpack_data_GPU_MWA8bit_transpose(const int nchan,const int ninp,const int batchsize, int log2_nchan,
                unsigned char *in_buf, cuFloatComplex *out_buf);
static float elapsed_time(struct timeval *start);
void printGPUDetails(FILE *fp);
int init_CMAC_1x1(void);
int init_CMAC_GxG(const int);

/* global vars */
int nchan=128;
int ninp=16;
int debug=1;
int naver=10000;
int fft_batchsize=25;
int wordtype=DATATYPE_COMPLEX_SIGNED_BYTE,wordsize=0,complex_data=1; // input data format type
int unpack_method=0;
int cmac_method=CMAC_GxG_4;
char *infilename=NULL,*outfilename=NULL;
int prod_type='B';  /* correlation product type: B: both, C: cross, A: auto */
//__device__ __constant__ uint2 inp_index_gpu[MAX_CORR_PROD]; // cannot fit for 256 inputs
__device__ uint2 inp_index_gpu[MAX_CORR_PROD];

int main(int argc, char * const argv[]) {
    size_t siz_inp_buf=0,siz_ft_buf=0,siz_corr_buf=0,siz_inp_gpu=0;
    int i,nmoves=0,iter=0,nav_written=0,log2_nchan;
    int filedone=0,ncorr;
    FILE *finp=NULL,*fout_ac=NULL,*fout_cc=NULL;
    char optstring[]="c:i:o:n:a:f:p:w:m:u:";
    unsigned char *staging_buf=NULL,*staging_buf_gpu=NULL;
    cufftComplex *inp_buf_gpu=NULL;
    cufftComplex *ft_buf=NULL;
    cuFloatComplex *corr_buf=NULL;
    cuFloatComplex *corr_buf_gpu=NULL;
    cudaError_t res;
    struct timeval thetime,starttime;
    float read_time=0,fft_time=0,cmac_time=0,sync_time=0,write_time=0,move_time=0,unpack_time=0;
    dim3 gridUnpack,blockUnpack;

    
    gettimeofday(&starttime,NULL);

    /* process command line args */
    if (argc <2) print_usage(argv);
    parse_cmdline(argc,argv,optstring);
    ncorr = ninp*(ninp+1)/2;

    /* check that nchan is power of 2 */
    for(i=1;i<30;i++) {
        if(1<<i == nchan) {
            log2_nchan=i;
            break;
        }
    }
    if (log2_nchan ==0) {
        fprintf(stderr,"ERROR: number of channels must be a power of 2\n");
    }

    /* calculate the size of various buffers in bytes */
    siz_inp_buf  = ninp*nchan*wordsize*fft_batchsize;
    siz_inp_gpu  = ninp*nchan*sizeof(cufftComplex)*fft_batchsize;
    siz_ft_buf   = ninp*nchan*sizeof(cufftComplex)*fft_batchsize;
    siz_corr_buf = ncorr*nchan*sizeof(cufftComplex);

    if (debug) {
        printGPUDetails(stderr);
        fprintf(stderr,"---\nNum inp:\t%d. Num corr products: %d\n",ninp,ncorr);
        fprintf(stderr,"CMAC method:\t%d\n",cmac_method);
        fprintf(stderr,"Num chan:\t%d\n",nchan);
        fprintf(stderr,"infile: \t%s\n",infilename);
        fprintf(stderr,"outfile:\t%s\n",outfilename);
        fprintf(stderr,"batchsize:\t%d\n",fft_batchsize);
        fprintf(stderr,"Input buffer size:\t%ld\n",(long)siz_inp_buf);
        //fprintf(stderr,"Size of uchar: %d\n",sizeof(unsigned char));
    }
    
    /* open input and output files */
    openFiles(infilename,outfilename,prod_type, &finp,&fout_ac,&fout_cc);

    /* allocate buffers */
    /* input buffer on host, which gets transferred to the GPU */
    res = cudaMallocHost((void **)&staging_buf,siz_inp_buf);
    if (res != 0) {
        fprintf(stderr,"failed to alloc host mem for staging_buf. Error: %s\n",cudaGetErrorString(res));
        exit(1);
    }
    
    /* staging buffer, which lives on the GPU. this holds packed (byte) data */
    res = cudaMalloc((void **)&staging_buf_gpu,siz_inp_buf);
    if (res != 0) {
        fprintf(stderr,"failed to alloc device mem for staging_buf_gpu\n");
        exit(1);
    }
    
    /* input buffer, which lives on the GPU. This holds unpacked (float) data */
    res = cudaMalloc((void **)&inp_buf_gpu,siz_inp_gpu);
    if (res != 0) {
        fprintf(stderr,"failed to alloc device mem for inp_buf_gpu\n");
        exit(1);
    }

    /* FFT result buffer, lives on GPU only */
    res = cudaMalloc((void **)&ft_buf,siz_ft_buf);
    if (res != 0) {
        fprintf(stderr,"failed to alloc device mem for ft_buf\n");
        exit(1);
    }

    /* for correlation products */
    /* results of CMAC on the GPU */
    res=cudaMalloc((void **)&corr_buf_gpu,siz_corr_buf);
    if (res != 0) {
        fprintf(stderr,"failed to alloc device mem for corr_buf_gpu\n");
        exit(1);
    }

    /* space on the host for CMAC results to be transferred to */
    res=cudaMallocHost((void **)&corr_buf,siz_corr_buf);
    if (res != 0) {
        fprintf(stderr,"failed to alloc host mem for corr_buf\n");
        exit(1);
    }

    /* init to zero, since stuff is accumulated into these arrays */
    cudaMemset(corr_buf_gpu,'\0',siz_corr_buf);
    cudaMemset(ft_buf,'\0',siz_ft_buf);
    cudaMemset(inp_buf_gpu,'\0',siz_inp_gpu);

    if ( ((nchan*ninp*fft_batchsize) % MAX_THREADS) != 0) {
        fprintf(stderr,"can't find integer mult of threads for unpack\n");
        exit(1);
    }

    /* execution configuration for the unpack/corner turn stage */
    if(unpack_method==0) {
        /* simple unpack method: each thread writes a time/channel for an input.
           Grid dimension X is over all inputs, then Y is for time */
        gridUnpack.x = ninp;
        gridUnpack.y = fft_batchsize;
        blockUnpack.x = nchan;
    }
    else {
        /* unpack using using shared memory, which makes reads and writes coalesced.
            Note that this is slower for older GPUs, as explained below */
        gridUnpack.x = ninp/TILE_SIZE_UNPACK;
        gridUnpack.y = fft_batchsize*nchan/TILE_SIZE_UNPACK;
        blockUnpack.x = TILE_SIZE_UNPACK;
        blockUnpack.y = UNPACK_BLOCK_ROWS;        
    }

    /* process file */
    while (!filedone) {
    
        /* read time chunk into buffers on host */
        gettimeofday(&thetime,NULL);
        if (readData(nchan, ninp,finp,staging_buf) !=0) {
            filedone=1;
        }
        read_time += elapsed_time(&thetime);
        
        if (!filedone) {

            /* wait for any running correlation threads to finish */
            gettimeofday(&thetime,NULL);
            cudaDeviceSynchronize();
            sync_time += elapsed_time(&thetime);
            if ( (res=cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr,"do_CMAC_gpu failed. Error: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }

            /* move new input to GPU */
            gettimeofday(&thetime,NULL);
            //if (debug) fprintf(stdout,"Moving batch %d to GPU\n",iter);
            res = cudaMemcpy(staging_buf_gpu,staging_buf,siz_inp_buf,cudaMemcpyHostToDevice);
            if (res != cudaSuccess) {
                fprintf(stderr,"Error on memcpy of data from host to device. Message: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }
            nmoves++;
            move_time += elapsed_time(&thetime);

            /* unpack the data into float format */
            gettimeofday(&thetime,NULL);
            switch (wordtype) {
                case DATATYPE_COMPLEX_SIGNED_BYTE:
                    unpack_data_GPU_16bit_simple<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize, (char2 *)staging_buf_gpu, inp_buf_gpu);
                    break;
                case DATATYPE_COMPLEX_UNSIGNED_BYTE:
                    unpack_data_GPU_uint8_complex_simple<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize, (char2 *)staging_buf_gpu, inp_buf_gpu);
                    break;
                case DATATYPE_MWA16:
                    if (unpack_method==1) {
                        unpack_data_GPU_MWA16bit_transpose<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize,log2_nchan,
                                         (unsigned short *)staging_buf_gpu, inp_buf_gpu);
                    }
                    else {
                        unpack_data_GPU_MWA16bit_simple<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize, (unsigned short *)staging_buf_gpu, inp_buf_gpu);
                    }
                    break;
                case DATATYPE_MWA8:
                    if (unpack_method==1) {
                        unpack_data_GPU_MWA8bit_transpose<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize,log2_nchan,
                                         (unsigned char *)staging_buf_gpu, inp_buf_gpu);
                    }
                    else {
                        unpack_data_GPU_MWA8bit_simple<<<gridUnpack,blockUnpack>>>(nchan,ninp,fft_batchsize, (unsigned char *)staging_buf_gpu, inp_buf_gpu);
                    }
                    break;
                default:
                    fprintf(stderr,"unknown data word type %d\n",wordtype);
                    exit(1);
            }
            cudaDeviceSynchronize(); /* must wait before starting FFT */
            unpack_time += elapsed_time(&thetime);            
            if ( (res=cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr,"unpack_data_GPU failed. Error: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }

            /* do the FFT */
            gettimeofday(&thetime,NULL);
            if (do_FFT_gpu(nchan,ninp,inp_buf_gpu,ft_buf) != CUFFT_SUCCESS) goto EXIT;
            cudaDeviceSynchronize();
            fft_time += elapsed_time(&thetime);

            /* do the CMAC. don't sync after this call since the next batch of data can be read in parallel. */
            gettimeofday(&thetime,NULL);
            do_CMAC_gpu(cmac_method,nchan,ninp,fft_batchsize, ft_buf, corr_buf_gpu);
            cmac_time += elapsed_time(&thetime);
            iter++;
        }
        
        /* write out if it is time to */
//        if ( (filedone && iter>0) || (iter*fft_batchsize >= naver) ) {
        if ( iter*fft_batchsize >= naver ) {
            cudaThreadSynchronize();
            gettimeofday(&thetime,NULL);
            /* fetch the accumulated results from the GPU */
            res=cudaMemcpy(corr_buf,corr_buf_gpu,siz_corr_buf,cudaMemcpyDeviceToHost);
            if (res != cudaSuccess) {
                fprintf(stderr,"Error on memcpy of results from device to host. Message: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }
 
            /* reset to zero, since stuff is accumulated into these arrays */
            res=cudaMemset(corr_buf_gpu,'\0',siz_corr_buf);
            if (res != cudaSuccess) {
                fprintf(stderr,"Error on memset on host. Message: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }

            gettimeofday(&thetime,NULL);
            writeOutput(fout_ac,fout_cc,ninp,nchan,iter,prod_type,corr_buf,1.0/(nchan*iter*fft_batchsize));
            if(debug) fprintf(stderr,"writing average of %d chunks\n",iter*fft_batchsize);
            iter=0;
            nav_written++;
            write_time += elapsed_time(&thetime);
        }
    }
    
    if (debug) {
        fprintf(stderr,"wrote %d averages. unused chunks: %d\n",nav_written,iter*fft_batchsize);
        fprintf(stderr,"Time reading:\t%g ms (done in parallel with CMAC)\n",read_time);
        fprintf(stderr,"Time moving:\t%g ms. N moves: %d, BW: %g GB/s\n",
                move_time,nmoves,(float)nmoves*(float)siz_inp_buf/(move_time*1e-3)*1e-9);
        fprintf(stderr,"Time unpacking:\t%g ms\n",unpack_time);
        fprintf(stderr,"Time FFTing:\t%g ms\n",fft_time);
        fprintf(stderr,"Time CMACing:\t%g ms (including read-time)\n",cmac_time+sync_time+read_time);
        fprintf(stderr,"Time writing:\t%g ms\n",write_time);
        fprintf(stderr,"Total time:\t%g ms\n",elapsed_time(&starttime));
    }

EXIT:
    /* clean up */
    fclose(finp);
    if(fout_ac !=NULL) fclose(fout_ac);
    if(fout_cc !=NULL) fclose(fout_cc);
    if (staging_buf_gpu != NULL) cudaFree(staging_buf_gpu);
    if (staging_buf != NULL) cudaFreeHost(staging_buf);
    if (ft_buf != NULL) cudaFree(ft_buf);
    if (inp_buf_gpu != NULL) cudaFree(inp_buf_gpu);
    if (corr_buf != NULL) cudaFreeHost(corr_buf);
    if (corr_buf_gpu != NULL) cudaFree(corr_buf_gpu);
    return 0;
}


/* unpack the byte data into float complex format on the GPU.
   The input is a sequence of n-byte words, one for each input for each time instant. input i, time t
   i.e. i0t0,i1t0,i2t0,i3t0,i0t1,i1t1,i2t1,i3t1,i0t2, etc.
   We need to re-order these into
   sequences for inputs in the FFT. i.e.
   i0t0,i0t1,...i0,t(nchan-1),i1,t0,i1,t1,...i1,t(nchan-1)
   least significant bits are reals, then imags
*/
/* unpack data - faster method with coalesced reads and writes
   this is for MWA 16-bit format (5+5 real, imag)
   This is based on the "transposeNew" project that comes with the CUDA SDK.
   read a block of TILE_SIZE_UNPACK*TILE_SIZE_UNPACK values into shared memory
   where threads read in the "Y" direction, then write the block into the result
   array where threads write in the "X" direction.
   
   Note that for older NVIDIA GPUs (version 1.0 and 1.1), this transpose is actually slower
   than the simple method. This is becuase (as far as I can tell), the reads are not actually
   coalesced because they are shorts or chars, not a 4-byte data type.
*/
__global__ void unpack_data_GPU_MWA16bit_transpose(const int nchan,const int ninp,const int batchsize, int log2_nchan,
                unsigned short *in_buf, cuFloatComplex *out_buf) {
    int i, batch=0, sample, real, imag, x_ind, y_ind, index_in, index_out,blk_y;
    int nchan_on_tilesize;
    __shared__ cuFloatComplex tile[TILE_SIZE_UNPACK][TILE_SIZE_UNPACK+1];

    //batch = blockIdx.y*TILE_SIZE_UNPACK/nchan;
    batch = (blockIdx.y*TILE_SIZE_UNPACK)>>log2_nchan;
    nchan_on_tilesize = nchan>>LOG2_TILESIZE;
    //blk_y = blockIdx.y%(nchan/TILE_SIZE_UNPACK); // use faster division for powers of 2 below.
    blk_y = blockIdx.y&(nchan_on_tilesize-1);
    
    // find array index for this thread in input array.    
    x_ind = blockIdx.x*TILE_SIZE_UNPACK + threadIdx.x;
    y_ind = blk_y*TILE_SIZE_UNPACK + threadIdx.y;
    index_in = batch*nchan*ninp + ninp*y_ind + x_ind;

    // likewise for output array
    x_ind = blk_y*TILE_SIZE_UNPACK + threadIdx.x;
    y_ind = blockIdx.x*TILE_SIZE_UNPACK + threadIdx.y;
    index_out = batch*nchan*ninp + nchan*y_ind + x_ind;

    for (i=0; i<TILE_SIZE_UNPACK ; i+=UNPACK_BLOCK_ROWS) {
        sample = in_buf[index_in + i*ninp];
        real = sample&DECODER_MASK;
        imag = (sample>>BITS_PER_NUM)&DECODER_MASK;
        if(imag >= TWOCOMP_ZEROPOINT) {
            imag -= TWOCOMP_MAXUINT;
        }
        if(real >= TWOCOMP_ZEROPOINT) {
            real -= TWOCOMP_MAXUINT;
        }
        tile[threadIdx.y+i][threadIdx.x] = make_cuFloatComplex(real,imag);
    }
    __syncthreads();
#ifdef __DEVICE_EMULATION__
    //printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, sample: %d, real: %d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample,real,imag); 
#endif
    for (i=0; i<TILE_SIZE_UNPACK ; i+=UNPACK_BLOCK_ROWS) {
        out_buf[index_out + i*nchan] = tile[threadIdx.x][threadIdx.y + i];
    }
}

/* unpack data - faster method with coalesced reads and writes
   this is for MWA 8-bit format (4+4 real, imag)
   This is based on the "transposeNew" project that comes with the CUDA SDK.
   read a block of TILE_SIZE_UNPACK*TILE_SIZE_UNPACK values into shared memory
   where threads read in the "Y" direction, then write the block into the result
   array where threads write in the "X" direction.
*/
__global__ void unpack_data_GPU_MWA8bit_transpose(const int nchan,const int ninp,const int batchsize, int log2_nchan,
                unsigned char *in_buf, cuFloatComplex *out_buf) {
    int i, batch=0, sample, real, imag, x_ind, y_ind, index_in, index_out,blk_y;
    int nchan_on_tilesize;
    __shared__ cuFloatComplex tile[TILE_SIZE_UNPACK][TILE_SIZE_UNPACK+1];

    //batch = blockIdx.y*TILE_SIZE_UNPACK/nchan;
    batch = (blockIdx.y*TILE_SIZE_UNPACK)>>log2_nchan;
    nchan_on_tilesize = nchan>>LOG2_TILESIZE;
    //blk_y = blockIdx.y%(nchan/TILE_SIZE_UNPACK); // use faster division for powers of 2 below.
    blk_y = blockIdx.y&(nchan_on_tilesize-1);
    
    // find array index for this thread in input array.    
    x_ind = blockIdx.x*TILE_SIZE_UNPACK + threadIdx.x;
    y_ind = blk_y*TILE_SIZE_UNPACK + threadIdx.y;
    index_in = batch*nchan*ninp + ninp*y_ind + x_ind;

    // likewise for output array
    x_ind = blk_y*TILE_SIZE_UNPACK + threadIdx.x;
    y_ind = blockIdx.x*TILE_SIZE_UNPACK + threadIdx.y;
    index_out = batch*nchan*ninp + nchan*y_ind + x_ind;

    for (i=0; i<TILE_SIZE_UNPACK ; i+=UNPACK_BLOCK_ROWS) {
        sample = in_buf[index_in + i*ninp];
        real = sample&DECODER_MASK_8BIT;
        imag = (sample>>BITS_PER_NUM_8BIT)&DECODER_MASK_8BIT;
        if(imag >= TWOCOMP_ZEROPOINT_8BIT) {
            imag -= TWOCOMP_MAXUINT_8BIT;
        }
        if(real >= TWOCOMP_ZEROPOINT_8BIT) {
            real -= TWOCOMP_MAXUINT_8BIT;
        }
        tile[threadIdx.y+i][threadIdx.x] = make_cuFloatComplex(real,imag);
    }
    __syncthreads();
#ifdef __DEVICE_EMULATION__
    //printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, sample: %d, real: %d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample,real,imag); 
#endif
    for (i=0; i<TILE_SIZE_UNPACK ; i+=UNPACK_BLOCK_ROWS) {
        out_buf[index_out + i*nchan] = tile[threadIdx.x][threadIdx.y + i];
    }
}


/* unpack data - simple method with non-coalesced reads, coalesced writes
   this is for MWA 16-bit format (5+5 real, imag)
*/
__global__ void unpack_data_GPU_MWA16bit_simple(const int nchan,const int ninp,const int batchsize, unsigned short *staging_buf,
                cuFloatComplex *inp_buf) {
    int out_index,inp_index;
    int sample,real,imag;
    
    inp_index = blockIdx.y*ninp*nchan + threadIdx.x*ninp + blockIdx.x;
    out_index = blockIdx.y*ninp*nchan + blockIdx.x*nchan + threadIdx.x;
    sample = staging_buf[inp_index];
    real = sample&DECODER_MASK;
    imag = (sample>>BITS_PER_NUM)&DECODER_MASK;
    if(imag >= TWOCOMP_ZEROPOINT) {
        imag -= TWOCOMP_MAXUINT;
    }
    if(real >= TWOCOMP_ZEROPOINT) {
        real -= TWOCOMP_MAXUINT;
    }
    
#ifdef __DEVICE_EMULATION__
    /*
    printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, sample: %d, real: 
%d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,s
ample,real,imag);
*/
     
#endif
    inp_buf[out_index] = make_cuFloatComplex(real,imag);
}

/* unpack data - simple method with non-coalesced reads, coalesced writes
   this is for raw 16-bit format (8+8 real, imag)
*/
__global__ void unpack_data_GPU_16bit_simple(const int nchan,const int ninp,const int batchsize, char2 *staging_buf,
                cuFloatComplex *inp_buf) {
    int out_index,inp_index;
    char2 sample;
    
    inp_index = blockIdx.y*ninp*nchan + threadIdx.x*ninp + blockIdx.x;
    out_index = blockIdx.y*ninp*nchan + blockIdx.x*nchan + threadIdx.x;
    sample = staging_buf[inp_index];
    
#ifdef __DEVICE_EMULATION__
    /*
    printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, real: 
%d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample.x,sample.y);
*/
     
#endif
    inp_buf[out_index] = make_cuFloatComplex(sample.x,sample.y);
}

/* unpack data - simple method with non-coalesced reads, coalesced writes
   this is for raw unsigned 8+8 real, imag
*/
__global__ void unpack_data_GPU_uint8_complex_simple(const int nchan,const int ninp,const int batchsize, char2 *staging_buf,
                cuFloatComplex *inp_buf) {
    int out_index,inp_index;
    char2 sample;
    
    inp_index = blockIdx.y*ninp*nchan + threadIdx.x*ninp + blockIdx.x;
    out_index = blockIdx.y*ninp*nchan + blockIdx.x*nchan + threadIdx.x;
    sample = staging_buf[inp_index];
    
#ifdef __DEVICE_EMULATION__
    /*
    printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, real: 
%d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample.x,sample.y);
*/
     
#endif
    inp_buf[out_index] = make_cuFloatComplex(sample.x-128.0,sample.y-128.0);
}




/* unpack data - simple method with non-coalesced reads, coalesced writes
   this is for MWA 8-bit format (4+4 real, imag)
*/
__global__ void unpack_data_GPU_MWA8bit_simple(const int nchan,const int ninp,const int batchsize, unsigned char *staging_buf,
                cuFloatComplex *inp_buf) {
    int out_index,inp_index;
    int sample,real,imag;
    
    inp_index = blockIdx.y*ninp*nchan + threadIdx.x*ninp + blockIdx.x;
    out_index = blockIdx.y*ninp*nchan + blockIdx.x*nchan + threadIdx.x;
    sample = staging_buf[inp_index];
    real = sample&DECODER_MASK_8BIT;
    imag = (sample>>BITS_PER_NUM_8BIT)&DECODER_MASK_8BIT;
    if(imag >= TWOCOMP_ZEROPOINT_8BIT) {
        imag -= TWOCOMP_MAXUINT_8BIT;
    }
    if(real >= TWOCOMP_ZEROPOINT_8BIT) {
        real -= TWOCOMP_MAXUINT_8BIT;
    }

#ifdef __DEVICE_EMULATION__
    printf("8bit blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, sample: %d, real: %d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample,(int)real,(int)imag);
#endif
    //inp_buf[out_index] = make_cuFloatComplex(real,imag);
    inp_buf[out_index] = make_cuFloatComplex((real+0.5),(imag+0.5));
}


/* wrapper function for GPU CMAC methods */
void do_CMAC_gpu(int method, const int nchan, const int ninp, const int batchsize,
            cufftComplex * const ft_buf, cufftComplex * const corr_buf) {
    static int initialised=0;
    
    switch(method) {
    
        case CMAC_1xG_2: {
            /* the CMAC is executed as a 2D grid of threads. The i axis has ninp
                blocks, the j axis has ninp/2 blocks .*/
            dim3 dimBlock(nchan);
            dim3 dimGrid(ninp,ninp/2);
            do_CMAC_gpu_1xG<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                                    (cuFloatComplex *) ft_buf,(cuFloatComplex *) corr_buf);
            break;
            }
        case CMAC_1xG_4: {
            /* the CMAC is executed as a 2D grid of threads. The i axis has ninp
                blocks, the j axis has ninp/4 blocks .*/
            dim3 dimBlock(nchan);
            dim3 dimGrid(ninp,ninp/4);
            do_CMAC_gpu_1xG_4<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                                    (cuFloatComplex *) ft_buf,(cuFloatComplex *) corr_buf);
            break;
            }
        case CMAC_1x1: {
            if (!initialised) {
                init_CMAC_1x1();
                initialised=1;
            }
            dim3 dimBlock(nchan);
            dim3 dimGrid(ninp*(ninp+1)/2);
            do_CMAC_gpu_1x1<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                                    (cuFloatComplex *) ft_buf,(cuFloatComplex *) corr_buf);
            
            break;
            }
        case CMAC_GxG_2: {
            if (!initialised) {
                init_CMAC_GxG(2);
                initialised=1;
            }
            dim3 dimBlock(MAX_CHAN/2,2);
            dim3 dimGrid(nchan/(MAX_CHAN/2),(ninp/2)*((ninp/2)+1)/2);
            do_CMAC_gpu_GxG<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                                    (cuFloatComplex *) ft_buf,(cuFloatComplex *) corr_buf);
            break;
            }
        case CMAC_GxG_4: {
            if (!initialised) {
                init_CMAC_GxG(4);
                initialised=1;
            }
            dim3 dimBlock(MAX_CHAN/4,4);
            dim3 dimGrid(nchan/(MAX_CHAN/4),(ninp/4)*((ninp/4)+1)/2);
            do_CMAC_gpu_GxG_4<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                                    (cuFloatComplex *) ft_buf,(cuFloatComplex *) corr_buf);
            break;
            }
        default: {
            fprintf(stderr,"do_CMAC_gpu: unknown CMAC method %d\n",method);
            exit(1);
            }
    }
}

 
/*  Do the CMAC on the GPU. At this stage, we have a contiguous buffer of dim [fft_batchsize][ninp][nchan]
    complex numbers in ft_buf. We need to accumulate these into correlation products in corr_buf.
    The output buffer is cuFloatComplex[ninp*(ninp+1)/2][nchan]
*/
/*
    The 1xG design uses shared memory to reduce the total number of global reads. The execution grid
    is ninp x (ninp/GRPSIZ) where GRPSIZ=2. The i index is the input index on the horizontal axis,
    the j index is the input index on the vertical axis and we want to compute the lower half of
    the correlation matrix. This means only compute products with i >= j.
    
    The kernel shares a value on the i axis between GRPSIZ (2) products on the j axis. This means that
    instead of 2*GRPSIZ reads, we do GRPSIZ+1 reads for GRPSIZ products.

    Shared memory is used to store the common value from the i index, registers are used to form the
    GRPSIZ accumulation products on the j index.
    
    The advantage of this approach over the group parallel approach is that all threads remain
    independent and no synchronisation points are required. 
*/
/* this version hard coded for GRPSIZ = 2*/
__global__ void do_CMAC_gpu_1xG(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {

    /* define inputs (a,b) and result (c) */
    __shared__ cuFloatComplex b[MAX_CHAN];
    cuFloatComplex c1,c0;
    int j = INTMULT(blockIdx.y,2);

    /* skip redundant input combinations. This is the same for all threads,
       so should be fast and inexpensive. */
    if (j > blockIdx.x) return;

    /* initialise accumulators */
    c1 = c0 = make_cuFloatComplex(0.0,0.0);

    /* loop over all the time indices (one for each FFT batch) since they all go into the same product */
    for(int batch_ind=0; batch_ind<batchsize; batch_ind++) {
        cuFloatComplex a;
        
        /* fetch the i index input, which is the same over the group */
        b[threadIdx.x] = cuConjf(ft_buf[INTMULT((INTMULT(batch_ind,ninp)+blockIdx.x),nchan) + threadIdx.x]);

        // fetch. j is always <= blockIdx.x for first product. 
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j),nchan) + threadIdx.x];
        c0 = cuCaddf(c0,cuCmulf(a,b[threadIdx.x]));

        // second product. might be redundant
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j+1),nchan) + threadIdx.x];
        c1 = cuCaddf(c1,cuCmulf(a,b[threadIdx.x]));
    }
    
    /* the following formula decodes the correlation index (including autocorrelations) from the input
       corr_index = blockIdx.y*(ninp+1) - blockIdx.y*(blockIdx.y+1)/2 + (blockIdx.x-blockIdx.y);
       indices x and y */
    int corr_index;

    // be careful with j*(j+1)/2. Need to do div by 2 last.
    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    // accumulate and store result 
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c0);
    
    // don't store redundant pairs
    if (++j > blockIdx.x) return;

    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c1);
}


/* same again, but for explicit G=4 */
__global__ void do_CMAC_gpu_1xG_4(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {

    /* define inputs (a,b) and result (c) */
    __shared__ cuFloatComplex b[MAX_CHAN];
    cuFloatComplex c[4];
    int j = INTMULT(blockIdx.y,4);

    /* skip redundant input combinations. This is the same for all threads,
       so should be fast and inexpensive. */
    if (j > blockIdx.x) return;

    /* initialise accumulators */
    c[0] = c[1] = c[2] = c[3] = make_cuFloatComplex(0.0,0.0);

    /* loop over all the time indices (one for each FFT batch) since they all go into the same product */
    for(int batch_ind=0; batch_ind<batchsize; batch_ind++) {
        cuFloatComplex a;
        
        /* fetch the i index input, which is the same over the group */
        b[threadIdx.x] = cuConjf(ft_buf[INTMULT((INTMULT(batch_ind,ninp)+blockIdx.x),nchan) + threadIdx.x]);

        // fetch j 
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j),nchan) + threadIdx.x];
        c[0] = cuCaddf(c[0],cuCmulf(a,b[threadIdx.x]));

        // second product. might be redundant
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j+1),nchan) + threadIdx.x];
        c[1] = cuCaddf(c[1],cuCmulf(a,b[threadIdx.x]));
        
        // third product. might be redundant
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j+2),nchan) + threadIdx.x];
        c[2] = cuCaddf(c[2],cuCmulf(a,b[threadIdx.x]));

        // fourth product. might be redundant
        a = ft_buf[INTMULT((INTMULT(batch_ind,ninp) + j+3),nchan) + threadIdx.x];
        c[3] = cuCaddf(c[3],cuCmulf(a,b[threadIdx.x]));
    }

    /* the following formula decodes the correlation index (including autocorrelations) from the input
       corr_index = blockIdx.y*(ninp+1) - blockIdx.y*(blockIdx.y+1)/2 + (blockIdx.x-blockIdx.y);
       indices x and y */
    int corr_index;

    // be careful with j*(j+1)/2. Need to do div by 2 last.
    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    // accumulate and store result 
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[0]);
    
    // don't store redundant pairs
    if (++j > blockIdx.x) return;

    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[1]);
    
    // don't store redundant pairs
    if (++j > blockIdx.x) return;

    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[2]);
    
    // don't store redundant pairs
    if (++j > blockIdx.x) return;

    corr_index = INTMULT(j,ninp) - INTMULT(j,(j+1))/2 + blockIdx.x;
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[3]);
}



/*  The GxG design uses shared memory to reduce the total number of global reads. The execution grid
    is (ninp/GRPSIZ) x (ninp/GRPSIZ) where GRPSIZ=2 or 4. The i index is the input index on the horizontal axis,
    the j index is the input index on the vertical axis and we want to compute the lower half of
    the correlation matrix. This means only compute products with i >= j.
    
    The kernel shares a value on the i axis between GRPSIZ products on the j axis. This means that
    instead of 2*GRPSIZ*GRPSIZ reads, we do 2*GRPSIZ read for GRPSIZ*GRPSIZ products.
    
    Each thread computes GRPSIZ correlation products. A thread reads a value on the i index and
    puts in shared memory to be used by itself and (GRPSIZ-1) other threads. It then reads

*/
/* this version hard coded for GRPSIZ =2 */
__global__ void do_CMAC_gpu_GxG(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {

    uint2 inp_ind;

    /* fetch the pre-calculated input indexes for this correlation product.
       this comes from constant cache, so is fast. */
    inp_ind = inp_index_gpu[blockIdx.y];

    /* shared i index data values */
    __shared__ cuFloatComplex d_i[2][MAX_CHAN/2];

    /* accumulators */
    cuFloatComplex c[2];
    c[0] = c[1] = make_cuFloatComplex(0.,0.);

    /* index of channel for this thread */
    const int chan = threadIdx.x + blockIdx.x*(MAX_CHAN/2);
    /* the start of the i,j indexes of interest are at inp_ind.x*GRPSIZ and inp_ind.y*GRPSIZ */
    const int j = inp_ind.y*2 + threadIdx.y; /* j index for thread */
    int       i = inp_ind.x*2 + threadIdx.y;

    /* loop over batches (successive time instants) */
    for (int batch=0; batch < batchsize; batch++) {
        /* data along j axis */
        cuFloatComplex d_j;

        /* fetch the input j value for this time instant */
        d_j = ft_buf[INTMULT(nchan,(INTMULT(ninp,batch) + j)) + chan];

        /* fetch the input i values for this time instant and conjugate */
        d_i[threadIdx.y][threadIdx.x] = cuConjf(ft_buf[INTMULT(nchan,(INTMULT(ninp,batch) + i)) + chan]);
        __syncthreads();

        /* do the CMAC */
        c[0] = cuCaddf(c[0],cuCmulf(d_i[0][threadIdx.x],d_j));
        c[1] = cuCaddf(c[1],cuCmulf(d_i[1][threadIdx.x],d_j));
        
        // another sync is necessary here so that all threads proceed to the next time batch together.
        __syncthreads();

#ifdef __DEVICE_EMULATION__
        if (batch==0) {
            printf("blk: %d,%d, thr: %d,%d, inps: %d,%d, i,j: %d,%d, chan: %d, d_j: (%g,%g), d_i: (%g,%g).",
                blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y, inp_ind.x, inp_ind.y, i,j,chan,
                cuCrealf(d_j),cuCimagf(d_j),
                cuCrealf(d_i[threadIdx.x][threadIdx.y]),cuCimagf(d_i[threadIdx.x][threadIdx.y]));
            printf(" c0: (%g,%g), c1: (%g,%g), c2: (%g,%g), c3: (%g,%g)\n",
                cuCrealf(c[0]),cuCimagf(c[0]),cuCrealf(c[1]),cuCimagf(c[1]),
                cuCrealf(c[2]),cuCimagf(c[2]),cuCrealf(c[3]),cuCimagf(c[3]));
        }
#endif
    }

    /* store accumulation results. Don't store redundant products. */
    /* the following formula decodes the correlation index (including autocorrelations) from the input
       indices i and j, for the conceptual bottom half of correlation matrix (i >= j products)
       corr_index = j*ninp - j*(j+1)/2 + i
       be careful with the divide by 2 not to cause rounding - don't factorize out j.
    */
    /* each thread has calculated 4 correlation products along the i axis, so we accumulate where
       i >= j */
    int corr_index;
    i = inp_ind.x*2; /* reset i back to start of 4x4 block */
    corr_index = j*ninp - j*(j+1)/2 + i;    /* calc the correlation index for lower triangle */
    corr_index = corr_index*nchan + chan;   /* turn this into array offset for the channel */

    for (i=0; i < 2; i++) {
        if (i+inp_ind.x*2 >= j) {
            corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[i]);
#ifdef __DEVICE_EMULATION__
            printf("storing products. blk: %d,%d, thr: %d,%d, i,j,chan: %d,%d,%d, cind: %d, arr_ind: %d, val: %g,%g\n",
                blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,i+inp_ind.x*2,j,chan,(corr_index-chan)/nchan,corr_index,
                (double)cuCrealf(corr_buf[corr_index]),(double)cuCimagf(corr_buf[corr_index]));
#endif
        }
        corr_index += nchan;
    }
#ifdef __DEVICE_EMULATION__
    fflush(stdout);
#endif
}


/* this version hard coded for G =4 */
__global__ void do_CMAC_gpu_GxG_4(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {

    uint2 inp_ind;

    /* fetch the pre-calculated input indexes for this correlation product.
       this comes from constant cache, so is fast. */
    inp_ind = inp_index_gpu[blockIdx.y];

    /* shared i index data values */
    __shared__ cuFloatComplex d_i[4][MAX_CHAN/4];

    /* accumulators */
    cuFloatComplex c[4];
    c[0] = c[1] = c[2] = c[3] = make_cuFloatComplex(0.,0.);

    /* index of channel for this thread */
    const int chan = threadIdx.x + blockIdx.x*(MAX_CHAN/4);
    /* the start of the i,j indexes of interest are at inp_ind.x*GRPSIZ and inp_ind.y*GRPSIZ */
    const int j = inp_ind.y*4 + threadIdx.y; /* j index for thread */
    int       i = inp_ind.x*4 + threadIdx.y;

    /* loop over batches (successive time instants) */
    for (int batch=0; batch < batchsize; batch++) {
        /* data along j axis */
        cuFloatComplex d_j;

        /* fetch the input i values for this time instant and conjugate */
        d_i[threadIdx.y][threadIdx.x] = cuConjf(ft_buf[INTMULT(nchan,(INTMULT(ninp,batch) + i)) + chan]);
        __syncthreads();

        /* putting this fetch second uses less registers for some reason */
        /* fetch the input j value for this time instant */
        d_j = ft_buf[INTMULT(nchan,(INTMULT(ninp,batch) + j)) + chan];

        /* do the CMAC */
        c[0] = cuCaddf(c[0],cuCmulf(d_i[0][threadIdx.x],d_j));
        c[1] = cuCaddf(c[1],cuCmulf(d_i[1][threadIdx.x],d_j));
        c[2] = cuCaddf(c[2],cuCmulf(d_i[2][threadIdx.x],d_j));
        c[3] = cuCaddf(c[3],cuCmulf(d_i[3][threadIdx.x],d_j));
        
        // another sync is necessary here so that all threads proceed to the next time batch together.
        __syncthreads();

#ifdef __DEVICE_EMULATION__
        if (batch==0) {
            printf("blk: %d,%d, thr: %d,%d, inps: %d,%d, i,j: %d,%d, chan: %d, d_j: (%g,%g), d_i: (%g,%g).",
                blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y, inp_ind.x, inp_ind.y, i,j,chan,
                cuCrealf(d_j),cuCimagf(d_j),
                cuCrealf(d_i[threadIdx.x][threadIdx.y]),cuCimagf(d_i[threadIdx.x][threadIdx.y]));
            printf(" c0: (%g,%g), c1: (%g,%g), c2: (%g,%g), c3: (%g,%g)\n",
                cuCrealf(c[0]),cuCimagf(c[0]),cuCrealf(c[1]),cuCimagf(c[1]),
                cuCrealf(c[2]),cuCimagf(c[2]),cuCrealf(c[3]),cuCimagf(c[3]));
        }
#endif
    }

    /* store accumulation results. Don't store redundant products. */
    /* the following formula decodes the correlation index (including autocorrelations) from the input
       indices i and j, for the conceptual bottom half of correlation matrix (i >= j products)
       corr_index = j*ninp - j*(j+1)/2 + i
       be careful with the divide by 2 not to cause rounding - don't factorize out j.
    */
    /* each thread has calculated 4 correlation products along the i axis, so we accumulate where
       i >= j */
    int corr_index;
    i = inp_ind.x*4; /* reset i back to start of 4x4 block */
    corr_index = j*ninp - j*(j+1)/2 + i;    /* calc the correlation index for lower triangle */
    corr_index = corr_index*nchan + chan;   /* turn this into array offset for the channel */

    for (i=0; i < 4; i++) {
        if (i+inp_ind.x*4 >= j) {
            corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c[i]);
#ifdef __DEVICE_EMULATION__
            printf("storing products. blk: %d,%d, thr: %d,%d, i,j,chan: %d,%d,%d, cind: %d, arr_ind: %d, val: %g,%g\n",
                blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,i+inp_ind.x*4,j,chan,(corr_index-chan)/nchan,corr_index,
                (double)cuCrealf(corr_buf[corr_index]),(double)cuCimagf(corr_buf[corr_index]));
#endif
        }
        corr_index += nchan;
    }
#ifdef __DEVICE_EMULATION__
    fflush(stdout);
#endif
}


/*  1x1 pair parallel design is the simplest CMAC method.
    ninp*(ninp+1)/2 thread blocks are started. The thread decodes its i and j index
    from the pre-calculated lookup table in inp_index_gpu. Each thread fetches
    the value for the i and j index and calculates the product, summing over several
    time intervals. The results is added into the accumulator in global memory at
    the end.
*/
__global__ void do_CMAC_gpu_1x1(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {

    /* inputs (a,b) and result (c) */
    cuFloatComplex c;
    uint2 inp_ind;

    /* fetch the pre-calculated input indexes for this correlation product.
       this comes from constant cache, so is fast. */
    inp_ind = inp_index_gpu[blockIdx.x];

    c = make_cuFloatComplex(0.0,0.0);

    /* loop over all the time indices (one for each FFT batch) since they all go into the same product */
    for(int batch_ind=0; batch_ind<batchsize; batch_ind++) {

        cuFloatComplex a,b;

        a = ft_buf[INTMULT(INTMULT(batch_ind,ninp)+inp_ind.x,nchan) + threadIdx.x];

        if (inp_ind.x==inp_ind.y) {
            // don't re-fetch for autocorrelations. It is slightly faster this way.
            b = cuConjf(a);
        } else {
            b = cuConjf(ft_buf[INTMULT(INTMULT(batch_ind,ninp)+inp_ind.y,nchan) + threadIdx.x]);
        }

#ifdef __DEVICE_EMULATION__
        //if (batch_ind==0) printf("blk: %d, thr: %d, cind: %d, batch: %d, inp1: %d, inp2: %d\n",
        //   blockIdx.x,threadIdx.x,blockIdx.x,batch_ind,(int)inp_ind.x,(int)inp_ind.y);
#endif
        /* do the complex mult (conjugation already done above) */
        c = cuCaddf(c,cuCmulf(a,b));

    }

    /* accumulate and store result */
    int corr_index = INTMULT(blockIdx.x,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c);
}


/* initialise the input index lookup table for 1x1 CMAC */
int init_CMAC_1x1(void) {
    int i,j,cindex=0;
    uint2 inp_index_cpu[MAX_CORR_PROD];
    cudaError_t res;

    /* zero entire array first */
    memset(inp_index_cpu, '\0', sizeof(inp_index_cpu));

    for(i=0; i<ninp; i++) {
        for (j=i; j<ninp; j++){
            inp_index_cpu[cindex] = make_uint2(i,j);
            cindex++;
        }
    }
    res = cudaMemcpyToSymbol(inp_index_gpu,inp_index_cpu,sizeof(inp_index_cpu));
    if (res != cudaSuccess) {
        fprintf(stderr,"Error on memcpy of inp_index to device. Message: %s\n",cudaGetErrorString(res));
        return res;
    }
    return 0;
}


/* initialise the input index lookup table for GxG CMAC */
int init_CMAC_GxG(const int grpsize) {
    int i,j,cindex=0;
    uint2 inp_index_cpu[MAX_CORR_PROD];
    cudaError_t res;
    
    /* zero entire array first */
    memset(inp_index_cpu, '\0', sizeof(inp_index_cpu));

    for(j=0; j<ninp/grpsize; j++) {
        for (i=j; i<ninp/grpsize; i++){
            inp_index_cpu[cindex] = make_uint2(i,j);
            cindex++;
        }
    }

    res = cudaMemcpyToSymbol(inp_index_gpu,inp_index_cpu,sizeof(inp_index_cpu));
    if (res != cudaSuccess) {
        fprintf(stderr,"Error on memcpy of inp_index to device. Message: %s\n",cudaGetErrorString(res));
        return res;
    }
    return 0;
}


/* execute the FFT on the GPU. The input and output buffer pointers must be for memory
   on the GPU device, not the host machine. FFTs are batched here together to avoid
   overhead of calling GPU and to maximise GPU utilisation. */
int do_FFT_gpu(int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf) {
    static cufftHandle plan=0;
    static int doneplan=0;

    cufftResult res;
    
    /* make the FFTW execution plans. The CUDA FFT can do a batch of 1D FFTs at the same time. (sweet!)
       memory must be a contiguous block the size of siz_inp_buf*fft_batchsize for input
       and ninp*(nchan)*sizeof(complex) for result */
    if (!doneplan) {
        
        res = cufftPlan1d(&plan,nchan,CUFFT_C2C,ninp*fft_batchsize);
        if (res != CUFFT_SUCCESS) {
            fprintf(stderr,"ERROR: cufftPlan1d failed with error code %d\n",res);
            return res;
        }
        if (debug) fprintf(stderr,"Made a plan (id: %d) for %d channel FFT with batch size %d*%d\n",
                        (int) plan,nchan,ninp,fft_batchsize);
        doneplan=1;
    }
    
    res = cufftExecC2C(plan,inp_buf,ft_buf,CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {
        fprintf(stderr,"cuda FFT failed with result code %d\n",res);
        return res;
    }
    return 0;
}

/* write out correlation products.
   Apply a normalisation factor that depends on the FFT length and the number
   of averages so the flux density is the same regardless of the spectral channel width
   NOTE: Complex input voltages contain negative and positive frequencies, so a spectrum
   of bandwidth B goes from -B/2 to B/2 freq. So we need to shift the output channel indices
   when writing out. In the FFT, channel 0 is the center of the band. Channel N/2 is the
   start of the band, going up to N, then wrapping around back to channel 0, up to N/2-1.
*/
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                cufftComplex *buf,float normaliser) {
    int inp1,inp2,cprod=0,chan,index;
    float *temp_buffer=NULL;

    temp_buffer = (float *)malloc(sizeof(float)*(nchan));

    for(inp1=0; inp1<ninp; inp1++) {
        for (inp2=inp1; inp2<ninp; inp2++) {
        	index = cprod*nchan;
            /* make an average by dividing by the number of chunks that went into the total */
            for (chan=0; chan<nchan; chan++) {
                buf[index+chan] = make_cuFloatComplex(cuCrealf(buf[index+chan])*normaliser,cuCimagf(buf[index+chan])*normaliser);
                /* convert the autocorrelation numbers into floats, since the imag parts will be zero*/
                if (inp1==inp2 && (prod_type == 'A' || prod_type=='B')){
                    temp_buffer[chan] = cuCrealf(buf[index+chan]);
                }
            }
            if(inp1==inp2 && (prod_type == 'A' || prod_type=='B')) {
                /* write the auto correlation product */
                fwrite(temp_buffer+nchan/2,sizeof(float),nchan/2,fout_ac);
                fwrite(temp_buffer,sizeof(float),nchan/2,fout_ac);
            }
            if(inp1!=inp2 && (prod_type == 'C' || prod_type=='B')) {
                /* write the cross correlation product */
                fwrite(buf+index+nchan/2,sizeof(cufftComplex),nchan/2,fout_cc);
                fwrite(buf+index,sizeof(cufftComplex),nchan/2,fout_cc);
            }

            /* reset the correlation products to zero */
            memset(buf+index,'\0',(nchan)*sizeof(cufftComplex));
            cprod++;
        }
    }
    if (temp_buffer!=NULL) free(temp_buffer);
}


/* incoming data is a stream of samples, one per input per time sample. This needs to be packed into arrays
   where nchan values for the same input channel are contiguous. Also, for FFT batching, we load fft_batchsize
   time chunks (1 chunk = nchan values per channel) at a time. This then means we have fft_batchsize sets of
   ninp sets of nchan samples per call to this function */
int readData(int nchan,int ninp,FILE *fpin,unsigned char *inp_buf) {
    int ntoread=0,nread;

    ntoread = ninp*nchan*fft_batchsize;
    
    nread = fread(inp_buf,wordsize,ninp*nchan*fft_batchsize,fpin);
    if(nread < ntoread) return 1;
            
    return 0;
}


/* open the input and output files */
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc) {
    char tempfilename[FILENAME_MAX];
    int res=0;
    //char *inputbuf=NULL;
    
    if (infilename == NULL) {
        fprintf(stderr,"No input file specified\n");
        exit(1);
    }
    if (outfilename == NULL) {
        fprintf(stderr,"No output file specified\n");
        exit(1);
    }
    
    /* sanity check: can only use stdout for one type of output */
    if((prod_type=='B') && strcmp(outfilename,"-")==0) {
        fprintf(stderr,"Can only use stdout for either auto or cross correlations, not both\n");
        exit(1);
    }
    
    /* check for special file name: "-", which indicates to use stdin/stdout */
    if (strcmp(infilename,"-")==0) {
        *fin = stdin;
    } else {
        *fin = fopen(infilename,"r");
        if (*fin ==NULL) {
            fprintf(stderr,"failed to open input file name: <%s>\n",infilename);
            exit(1);
        }
    }
    /* if input is a pipe, see if we can increase its size */
    res=fcntl(fileno(*fin),F_GETPIPE_SZ,0);
    if (res > 0) {
        if (debug) {
            fprintf(stderr,"Current input pipe size: %d\n",res);
        }
        res=fcntl(fileno(*fin),F_SETPIPE_SZ,1048576);
        res=fcntl(fileno(*fin),F_GETPIPE_SZ,0);
         if (debug) {
            fprintf(stderr,"New input pipe size: %d\n",res);
        }
    }
    
    if ((prod_type=='A') && strcmp(outfilename,"-")==0) {
        *fout_ac = stdout;
    } else if ((prod_type=='C') && strcmp(outfilename,"-")==0) {
        *fout_cc = stdout;
    } else {
        if (prod_type=='A' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LACSPC");
            *fout_ac = fopen(tempfilename,"w");
            if (*fout_ac ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
        if (prod_type=='C' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LCCSPC");
            *fout_cc = fopen(tempfilename,"w");
            if (*fout_cc ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
    }
    
    return 0;
}


void parse_cmdline(int argc, char * const argv[], const char *optstring) {
    int c;
    
    while ((c=getopt(argc,argv,optstring)) != -1) {
        switch(c) {
            case 'c':
                nchan = atoi(optarg);
                if (nchan <=0 || nchan > MAX_CHAN || nchan %8 !=0) {
                    fprintf(stderr,"bad number of channels: %d. Max: %d. Must be power of 2\n",nchan,MAX_CHAN);
                    print_usage(argv);
                }
                break;
            case 'n':
                ninp = atoi(optarg);
                if (ninp <=0 || ninp > MAX_INPUTS) {
                    fprintf(stderr,"bad number of inputs: %d\n",ninp);
                    print_usage(argv);
                }
                break;
            case 'a':
                naver = atoi(optarg);
                if (naver <=0 || naver > 1000000) {
                    fprintf(stderr,"bad number of averages: %d\n",naver);
                    print_usage(argv);
                }
                break;
            case 'm':
                cmac_method = atoi(optarg);
                if (cmac_method < 0 || cmac_method >= CMAC_END) {
                    fprintf(stderr,"Unknown CMAC method: %d\n",cmac_method);
                    print_usage(argv);
                }
                break;
            case 'f':
                fft_batchsize = atoi(optarg);
                if (fft_batchsize <=0 || fft_batchsize > 1024) {
                    fprintf(stderr,"bad fft_batchsize: %d\n",naver);
                    print_usage(argv);
                }
                break;
            case 'i':
                infilename=optarg;
                break;
            case 'o':
                outfilename=optarg;
                break;
            case 'w':
                wordtype=atoi(optarg);
                if (wordtype < 0 || wordtype > DATATYPE_COMPLEX_SIGNED_BYTE) {
                    fprintf(stderr,"Bad data word type: %d\n",wordtype);
                    print_usage(argv);
                }
                if (wordtype==DATATYPE_MWA16)          wordsize=2;
                if (wordtype==DATATYPE_MWA8)           wordsize=1;
                if (wordtype==DATATYPE_COMPLEX_SHORT)  wordsize=4;
                if (wordtype==DATATYPE_COMPLEX_FLOAT)  wordsize=8;
                if (wordtype==DATATYPE_COMPLEX_UNSIGNED_BYTE || 
                    wordtype==DATATYPE_COMPLEX_SIGNED_BYTE) wordsize=2;
                if (wordtype==DATATYPE_UNSIGNED_BYTE || wordtype==DATATYPE_SIGNED_BYTE) {
                    wordsize=1;
                    complex_data=0;
                }
                if (wordtype==DATATYPE_UNSIGNED_SHORT|| wordtype==DATATYPE_SIGNED_SHORT) {
                    wordsize=2;
                    complex_data=0;
                }
                if (wordtype==DATATYPE_FLOAT) {
                    wordsize=4;
                    complex_data=0;
                }
                if (wordtype==DATATYPE_REAL_UNSIGNED_4BIT) {
                    fprintf(stderr,"Data word type: %d not supported\n",wordtype);
                    complex_data=0;
                    exit(1);
                }

                break;
            case 'u':
                unpack_method=atoi(optarg);
                if (unpack_method < 0 || unpack_method > 1) {
                    fprintf(stderr,"Bad unpack method: %d\n",unpack_method);
                    print_usage(argv);
                }
                break;
            case 'p':
                prod_type = toupper(optarg[0]);
                if (prod_type!='A' && prod_type !='B' && prod_type != 'C') {
                    fprintf(stderr,"bad correlation product type: %c\n",prod_type);
                    print_usage(argv);
                }
                break;
            default:
                fprintf(stderr,"unknown option %c\n",c);
                print_usage(argv);
        }
    }
    if (wordsize==0) {
        fprintf(stderr,"wordsize is 0\n");
        exit(1);
    }
}

/* returns the elapsed wall-clock time, in ms, since start (without resetting start) */
static float elapsed_time(struct timeval *start){
    struct timeval now;
    gettimeofday(&now,NULL);
    return 1.e3f*(float)(now.tv_sec-start->tv_sec) +
        1.e-3f*(float)(now.tv_usec-start->tv_usec);
}


void printGPUDetails(FILE *fp) {
    cudaError_t res;
    int numdev=0;
    struct cudaDeviceProp devprop;
    
    res = cudaGetDeviceCount(&numdev);
    if (res != 0) {
        fprintf(fp,"failed to get number of CUDA devices\n");
        exit(1);
    }
    fprintf(fp,"There are %d devices.\n",numdev);
    
    for(int dev=0; dev< numdev; dev++) {
        res = cudaGetDeviceProperties(&devprop,dev);
        if (res != 0) {
            fprintf(fp,"failed to get properties for device %d\n",dev);
            exit(1);
        }
    
        fprintf(fp,"Device %d:\nname:\t\t%s\n",dev,devprop.name);
        fprintf(fp,"MEM:\t\t%ld MB\n",(long)(devprop.totalGlobalMem)/(1024*1024));
        fprintf(fp,"Shmem/block:\t%ld\n",(long)devprop.sharedMemPerBlock);
        fprintf(fp,"regs/block:\t%ld\n",(long)devprop.regsPerBlock);
        fprintf(fp,"Warp size:\t%d\n",devprop.warpSize);
        fprintf(fp,"Mem pitch:\t%ld\n",(long)devprop.memPitch);
        fprintf(fp,"Max thr/blk:\t%d\n",devprop.maxThreadsPerBlock);
        fprintf(fp,"Max dim/blk:\t%d,%d,%d\n",devprop.maxThreadsDim[0],devprop.maxThreadsDim[1],devprop.maxThreadsDim[2]);
        fprintf(fp,"Tot const mem:\t%ld\n",(long)devprop.totalConstMem);
        fprintf(fp,"Version:\t%d.%d\n",devprop.major,devprop.minor);
        fprintf(fp,"clockrate:\t%d\n",devprop.clockRate);
        fprintf(fp,"texture algn:\t%ld\n",(long)devprop.textureAlignment);
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-p type\t\tspecify correlation product type(s). A: auto, C: cross, B: both. default: %c\n",prod_type);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels. default: %d\n",nchan);
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. default: %d\n",ninp);
    fprintf(stderr,"\t-a num\t\tspecify number of averages before output. default: %d\n",naver);
    fprintf(stderr,"\t-f num\t\tspecify fft batchsize. default: %d\n",fft_batchsize);
    fprintf(stderr,"\t-w num\t\tspecify data word type. Default: %d\n",wordtype);
    fprintf(stderr,"\t      \t%d: MWA 5+5bit real/imag encoded in 16 bits.\n",DATATYPE_MWA16);
    fprintf(stderr,"\t      \t%d: MWA 4+4bit real/imag encoded in 8 bits.\n",DATATYPE_MWA8);
    fprintf(stderr,"\t      \t%d: Native 16+16 bit signed int real/imag.\n",DATATYPE_COMPLEX_SHORT);
    fprintf(stderr,"\t      \t%d: Native 32+32 bit float complex real/imag.\n",DATATYPE_COMPLEX_FLOAT);
    fprintf(stderr,"\t      \t%d: Signed byte (real only).\n",DATATYPE_SIGNED_BYTE);
    fprintf(stderr,"\t      \t%d: Unsigned byte (real only).\n",DATATYPE_UNSIGNED_BYTE);
    fprintf(stderr,"\t      \t%d: Signed short (real only),\n",DATATYPE_SIGNED_SHORT);
    fprintf(stderr,"\t      \t%d: Unsigned short (real only),\n",DATATYPE_UNSIGNED_SHORT);
    fprintf(stderr,"\t      \t%d: Single precision float (real only),\n",DATATYPE_FLOAT);
    fprintf(stderr,"\t      \t%d: Native 8+8 bit unsigned int (complex),\n",DATATYPE_COMPLEX_UNSIGNED_BYTE);
    fprintf(stderr,"\t      \t%d: Native 8+8 bit signed int (complex),\n",DATATYPE_COMPLEX_SIGNED_BYTE);
    fprintf(stderr,"\t-u num\t\tspecify data unpack & corner turn method. (0=standard, 1=transpose) Default: %d\b\n",unpack_method);
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");
    fprintf(stderr,"\t-m num\t\tspecify CMAC method. 0: 1x1, 1: 1xG=2, 2: 1xG=4, 3: GxG=2, 4: GxG=4. Default: %d\n",cmac_method);
    exit(0);
}

