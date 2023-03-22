/* experimental GPU-based correlation code for a small (<=64) input
 * sampling system to run on a workstation class computer. There is an
 * upper limit to the number of supported channels due to the max number
 * of threads that can be on the GPU. This is currently 256.
 *
 * Author: Randall Wayth. October, 2007.
 *
 * to compile with CUDA: nvcc -D_CU_USE_NATIVE_COMPLEX -o corr_gpu_complex corr_gpu_complex.cu -lcufft
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

#define BUFFER_SIZE (1024*64)

#define MAX_THREADS 512
#define MAX_INPUTS 64
#define MAX_CORR_PROD (MAX_INPUTS*(MAX_INPUTS+1)/2)
#define MAX_CHAN 512
#define INTMULT(a,b) __mul24((a),(b))

#define BITS_PER_NUM 5
#define TWOCOMP_MAXUINT   (1<<(BITS_PER_NUM))    // = 32 for 5 bits
#define TWOCOMP_ZEROPOINT (1<<(BITS_PER_NUM-1))  // = 16 for 5 bits
#define DECODER_MASK (TWOCOMP_MAXUINT -1)

#define BITS_PER_NUM_8BIT 4
#define TWOCOMP_MAXUINT_8BIT   (1<<(BITS_PER_NUM_8BIT))    // = 16 for 4 bits
#define TWOCOMP_ZEROPOINT_8BIT (1<<(BITS_PER_NUM_8BIT-1))  // = 8  for 4 bits
#define DECODER_MASK_8BIT (TWOCOMP_MAXUINT_8BIT -1)

/* function prototypes */
void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[], const char *optstring);
int readData(int nchan,int ninp,FILE *fpin,unsigned char *inp_buf);
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc);
int do_FFT_gpu(int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf);
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                cuFloatComplex *buf,float normaliser);
__global__ void do_CMAC_gpu(const int nchan, const int ninp, const int batchsize,
         cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf);
__global__ void unpack_data_GPU(const int nchan,const int ninp,const int batchsize, const int ispowof2,
                        const int log2_nchan_ninp,const int log2_nchan, unsigned short *staging_buf,
                        cufftComplex *inp_buf_gpu);
__global__ void unpack_data_GPU_8bit(const int nchan,const int ninp,const int batchsize, const int ispowof2,
                        const int log2_nchan_ninp,const int log2_nchan, unsigned char *staging_buf,
                        cufftComplex *inp_buf_gpu);
static float elapsed_time(struct timeval *start);
void printGPUDetails(FILE *fp);

/* global vars */
int nchan=128;
int ninp=16;
int debug=1;
int naver=99999;
int fft_batchsize=32;
int wordsize=2; // bytes per data word
char *infilename=NULL,*outfilename=NULL;
int prod_type='B';  /* correlation product type: B: both, C: cross, A: auto */

int main(int argc, char * const argv[]) {
    size_t siz_inp_buf=0,siz_ft_buf=0,siz_corr_buf=0,siz_inp_gpu=0;
    int i,nmoves=0,iter=0,nav_written=0;
    short filedone=0,ncorr,inp_is_pow_2=0,log2_nchan_ninp=0,log2_nchan=0;
    FILE *finp=NULL,*fout_ac=NULL,*fout_cc=NULL;
    char optstring[]="c:i:o:n:a:f:p:w:";
    unsigned char *staging_buf=NULL,*staging_buf_gpu=NULL;
    cufftComplex *inp_buf_gpu=NULL;
    cufftComplex *ft_buf=NULL;
    cuFloatComplex *corr_buf=NULL;
    cuFloatComplex *corr_buf_gpu=NULL;
    cudaError_t res;
    struct timeval thetime,starttime;
    float read_time=0,fft_time=0,cmac_time=0,sync_time=0,write_time=0,move_time=0,unpack_time=0;
    
    gettimeofday(&starttime,NULL);

    /* process command line args */
    if (argc <2) print_usage(argv);
    parse_cmdline(argc,argv,optstring);
    ncorr = ninp*(ninp+1)/2;

    /* check that nchan is power of 2 */
    for(i=1;i<20;i++) {
        if(1<<i == nchan) {
            log2_nchan=i;
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
        fprintf(stderr,"Num chan:\t%d\n",nchan);
        fprintf(stderr,"infile: \t%s\n",infilename);
        fprintf(stderr,"outfile:\t%s\n",outfilename);
        fprintf(stderr,"batchsize:\t%d\n",fft_batchsize);
        fprintf(stderr,"Input buffer size:\t%d\n",siz_inp_buf);
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

    /* calculate the execution configuration for the CMAC on the GPU */
    /* the CMAC is executed as a set of 1D threads. */
    dim3 dimBlock(nchan);
    dim3 dimGrid(ninp,ninp);

    if ( ((nchan*ninp*fft_batchsize) % MAX_THREADS) != 0) {
        fprintf(stderr,"can't find integer mult of threads for unpack\n");
        exit(1);
    }

    //dim3 dimBlockUnpack(MAX_THREADS);
    //dim3 dimGridUnpack(2*nchan*ninp*fft_batchsize/MAX_THREADS);
    dim3 dimBlockUnpack(nchan);
    dim3 dimGridUnpack(ninp*fft_batchsize);

    /* if nchan*ninp is a power of two, set a flag. This makes unpacking faster */
    for(i=1;i<20;i++) {
        if(1<<i == ninp*nchan) {
            inp_is_pow_2=1;
            log2_nchan_ninp=i;
        }
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
            cudaThreadSynchronize();
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
            if (wordsize==2) {
                unpack_data_GPU<<<dimGridUnpack,dimBlockUnpack>>>(nchan,ninp,fft_batchsize,inp_is_pow_2,log2_nchan_ninp,
                            log2_nchan, (unsigned short *)staging_buf_gpu, inp_buf_gpu);
            }
            else{
                unpack_data_GPU_8bit<<<dimGridUnpack,dimBlockUnpack>>>(nchan,ninp,fft_batchsize,inp_is_pow_2,log2_nchan_ninp,
                            log2_nchan, (unsigned char *)staging_buf_gpu, inp_buf_gpu);
            }
            cudaThreadSynchronize(); /* must wait before starting FFT */
            unpack_time += elapsed_time(&thetime);            
            if ( (res=cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr,"unpack_data_GPU failed. Error: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }

            /* do the FFT */
            gettimeofday(&thetime,NULL);
            if (do_FFT_gpu(nchan,ninp,inp_buf_gpu,ft_buf) != CUFFT_SUCCESS) goto EXIT;
            cudaThreadSynchronize();
            fft_time += elapsed_time(&thetime);

            /* do the CMAC. don't sync after this call since the next batch of data can be read in parallel. */
            gettimeofday(&thetime,NULL);
            do_CMAC_gpu<<<dimGrid,dimBlock>>>(nchan,ninp,fft_batchsize,
                        (cuFloatComplex *)ft_buf, (cuFloatComplex *) corr_buf_gpu);
            cmac_time += elapsed_time(&thetime);
        }
        
        /* write out if it is time to */
        if (filedone || (++iter)*fft_batchsize >= naver) {
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
   The input is a sequence of 2-byte words, one for each input for each time instant. input i, time t
   i.e. i0t0,i1t0,i2t0,i3t0,i0t1,i1t1,i2t1,i3t1,i0t2, etc.
   We need to re-order these into
   sequences for inputs in the FFT. i.e.
   i0t0,i0t1,...i0,t(nchan-1),i1,t0,i1,t1,...i1,t(nchan-1)
   least significant bits are reals, then imags
*/
__global__ void unpack_data_GPU(const int nchan,const int ninp,const int batchsize, const int ispowof2,
                const int log2_nchan_ninp,const int log2_nchan, unsigned short *staging_buf,
                cufftComplex *inp_buf) {
    int output_index,batch_index,intra_batch_index,input_index,time_index,sample_index;
    short sample,real,imag;
    
    /* calculate the index in the destination array */
    output_index = blockIdx.x*nchan + threadIdx.x;
    
    /* as per CUDA manual, integer division and modulo are very expensive, so use bitwise operations instead
       which are valid for power-of-2 divisors if possible */
    if (ispowof2) {
        batch_index = output_index>>log2_nchan_ninp;
        intra_batch_index = output_index&(nchan*ninp -1); // valid only for power-of-two nchan*ninp
        input_index = intra_batch_index>>log2_nchan;
        time_index  = intra_batch_index&(nchan -1); // valid for power-of-two nchan
        sample_index= input_index + ninp*(time_index + batch_index*nchan);
    } else {
        batch_index = output_index/(nchan*ninp);
        intra_batch_index = output_index%(nchan*ninp);
        //intra_batch_index = output_index&(nchan*ninp -1); // valid only for power-of-two nchan*ninp
        input_index = intra_batch_index/(nchan);
        time_index  = intra_batch_index&(nchan -1); // valid for power-of-two nchan
        sample_index= input_index + ninp*(time_index + batch_index*nchan);
    }
    sample = staging_buf[sample_index];
    real = sample&DECODER_MASK;
    imag = (sample>>BITS_PER_NUM)&DECODER_MASK;
    if(imag >= TWOCOMP_ZEROPOINT) {
        imag -= TWOCOMP_MAXUINT;
    }
    if(real >= TWOCOMP_ZEROPOINT) {
        real -= TWOCOMP_MAXUINT;
    }
    
#ifdef __DEVICE_EMULATION__
    //printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d, sample: %d, real: %d, imag: %d\n", blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index,sample,real,imag);
     
#endif
    inp_buf[output_index] = make_cuFloatComplex(real,imag);
    //inp_buf[output_index] = make_cuFloatComplex(real+0.5,imag+0.5);
}



/* unpack the byte data into float complex format on the GPU.
   The input is a sequence of 1-byte words, one for each input for each time instant. input i, time t
   i.e. i0t0,i1t0,i2t0,i3t0,i0t1,i1t1,i2t1,i3t1,i0t2, etc.
   We need to re-order these into
   sequences for inputs in the FFT. i.e.
   i0t0,i0t1,...i0,t(nchan-1),i1,t0,i1,t1,...i1,t(nchan-1)
*/
__global__ void unpack_data_GPU_8bit(const int nchan,const int ninp,const int batchsize, const int ispowof2,
                const int log2_nchan_ninp,const int log2_nchan, unsigned char *staging_buf,
                cufftComplex *inp_buf) {
    int output_index,batch_index,intra_batch_index,input_index,time_index,sample_index;
    char sample,real,imag;
    
    /* calculate the index in the destination array */
    output_index = blockIdx.x*nchan + threadIdx.x;
    
    /* as per CUDA manual, integer division and modulo are very expensive, so use bitwise operations instead
       which are valid for power-of-2 divisors if possible */
    if (ispowof2) {
        batch_index = output_index>>log2_nchan_ninp;
        intra_batch_index = output_index&(nchan*ninp -1); // valid only for power-of-two nchan*ninp
        input_index = intra_batch_index>>log2_nchan;
        time_index  = intra_batch_index&(nchan -1); // valid for power-of-two nchan
        sample_index= input_index + ninp*(time_index + batch_index*nchan);
    } else {
        batch_index = output_index/(nchan*ninp);
        intra_batch_index = output_index%(nchan*ninp);
        //intra_batch_index = output_index&(nchan*ninp -1); // valid only for power-of-two nchan*ninp
        input_index = intra_batch_index/(nchan);
        time_index  = intra_batch_index&(nchan -1); // valid for power-of-two nchan
        sample_index= input_index + ninp*(time_index + batch_index*nchan);
    }
    sample = staging_buf[sample_index];
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
    //inp_buf[output_index] = make_cuFloatComplex(real,imag);
    inp_buf[output_index] = make_cuFloatComplex((real+0.5),(imag+0.5));
}


 
/*  Do the CMAC on the GPU. At this stage, we have a contiguous buffer of dim [fft_batchsize][ninp][nchan]
    complex numbers in ft_buf.
    We need to accumulate these into correlation products in corr_buf. nchan threads are executed, each
    of them reading consecutive memory locations. This way we get all channels in the FFTd spectrum including
    the DC term.
*/
/*  this function is where most of the action happens. We want to guarantee coalesced memory
    reads across threads. This means aligning the arrays correctly. The output array is
    nchan*ncorr, so the beginning of each nchan sub-array correlation product will always be aligned
    for power-of-two FFT sizes. This guarantees colasced writes to the output array.
    To guarantee coalesced reads, the first (DC) component of FFTd spectrum is read. So, a total of nchan
    threads are required.
*/
__global__ void do_CMAC_gpu(const int nchan, const int ninp, const int batchsize,
            cuFloatComplex * const ft_buf, cuFloatComplex * const corr_buf) {


    /* skip redundant input combinations. This is the same for all threads, so is fast
       and relatively inexpensive. */
    if (blockIdx.y > blockIdx.x) return;

#ifdef __DEVICE_EMULATION__
#endif

    /* define inputs (a,b) and result (c) */
    cuFloatComplex c = make_cuFloatComplex(0.0,0.0);

    /* loop over all the time indices (one for each FFT batch) since they all go into the same product */
    for(int batch_ind=0; batch_ind<batchsize; batch_ind++) {

        cuFloatComplex a,b;

        a = ft_buf[INTMULT(INTMULT(batch_ind,ninp)+blockIdx.y,nchan) + threadIdx.x];

        if (blockIdx.x==blockIdx.y) {
            // don't re-fetch for autocorrelations
            b = cuConjf(a);
        } else {
            b = cuConjf(ft_buf[INTMULT(INTMULT(batch_ind,ninp)+blockIdx.x,nchan) + threadIdx.x]);
        }

        
#ifdef __DEVICE_EMULATION__
        //if (batch_ind==0) printf("blk: %d, thr: %d, cind: %d, batch: %d, inp1: %d, inp2: %d\n",
        //   blockIdx.x,threadIdx.x,blockIdx.x,batch_ind,(int)inp_ind.x,(int)inp_ind.y);
#endif
        /* do the complex mult (conjugation above). It makes no difference to write out the
           complex mult explicily here - I tried. */
        c = cuCaddf(c,cuCmulf(a,b));
    }
    /* the following formula decodes the correlation index (including autocorrelations) from the input
//    int corr_index = blockIdx.y*(ninp+1) - blockIdx.y*(blockIdx.y+1)/2 + (blockIdx.x-blockIdx.y);
       indices x and y */
    int corr_index = INTMULT(blockIdx.y,ninp) - INTMULT(blockIdx.y,(blockIdx.y+1))/2 + blockIdx.x;
    /* accumulate and store result */
    corr_index = INTMULT(corr_index,nchan) + threadIdx.x;
    corr_buf[corr_index] = cuCaddf(corr_buf[corr_index],c);
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
        /* make an extra special input buffer for stdin */
        //inputbuf = (char *)malloc(BUFFER_SIZE);
        //setvbuf(stdin, inputbuf, _IOFBF , BUFFER_SIZE);
    } else {
        *fin = fopen(infilename,"r");
        if (*fin ==NULL) {
            fprintf(stderr,"failed to open input file name: <%s>\n",infilename);
            exit(1);
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
                wordsize=atoi(optarg);
                if (wordsize < 1 || wordsize > 2) {
                    fprintf(stderr,"Bad data word size: %d\n",wordsize);
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
        fprintf(fp,"MEM:\t\t%d MB\n",(devprop.totalGlobalMem)/(1024*1024));
        fprintf(fp,"Shmem per block:\t%d\n",devprop.sharedMemPerBlock);
        fprintf(fp,"regs  per block:\t%d\n",devprop.regsPerBlock);
        fprintf(fp,"Warp size:\t%d\n",devprop.warpSize);
        fprintf(fp,"Mem pitch:\t%d\n",devprop.memPitch);
        fprintf(fp,"Max thr per blk:\t%d\n",devprop.maxThreadsPerBlock);
        fprintf(fp,"Max dim per blk:\t%d,%d,%d\n",devprop.maxThreadsDim[0],devprop.maxThreadsDim[1],devprop.maxThreadsDim[2]);
        fprintf(fp,"Tot const mem:  \t%d\n",devprop.totalConstMem);
        fprintf(fp,"Version:\t%d.%d\n",devprop.major,devprop.minor);
        fprintf(fp,"clockrate:\t%d\n",devprop.clockRate);
        fprintf(fp,"texture algn:\t%d\n",devprop.textureAlignment);
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-p type\t\tspecify correlation product type(s). A: auto, C: cross, B: both. default: %c\n",prod_type);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels. default: %d\n",nchan);
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. default: %d\n",ninp);
    fprintf(stderr,"\t-a num\t\tspecify number of averages before output. default: %d\n",naver);
    fprintf(stderr,"\t-f num\t\tspecify fft batchsize. default: %d\n",fft_batchsize);
    fprintf(stderr,"\t-w num\t\tspecify data word size in bytes (1 or 2). Default: %d\n",wordsize);
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");
    exit(0);
}
