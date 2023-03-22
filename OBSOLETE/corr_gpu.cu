/* experimental GPU-based correlation code for a small (<=64) input
 * sampling system to run on a workstation class computer. There is an
 * upper limit to the number of supported channels due to the max number
 * of threads that can be on the GPU. This is currently 256.
 *
 * Author: Randall Wayth. October, 2007.
 *
 * to compile with CUDA: nvcc -D_CU_USE_NATIVE_COMPLEX -o corr_gpu corr_gpu.cu -lcufft
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "/usr/include/complex.h"
#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#define MAX_THREADS 512
#define MAX_INPUTS 64
#define MAX_CORR_PROD (MAX_INPUTS*(MAX_INPUTS+1)/2)
#define MAX_CHAN 512
#define SIZE_INP_DATA 1  /* bytes per input sample */

/* function prototypes */
void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[], const char *optstring);
int readData(int nchan,int ninp,FILE *fpin,unsigned char *inp_buf);
int openFiles(char *infilename, char *outfilename, FILE **fin, FILE **fout);
int do_FFT_gpu(int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf);
void writeOutput(FILE *fout,int ncorr, int nchan, const float normaliser, cuFloatComplex *buf);
__global__ void do_CMAC_gpu(const int nchan, const int ninp, const int ncorr, const int batchsize,
        const int2 *inp_index, cuFloatComplex *ft_buf, cuFloatComplex *corr_buf);
__global__ void unpack_data_GPU(const int nchan,const int ninp,const int batchsize, int log2_2nchan, unsigned char *staging_buf,
                        cufftComplex *inp_buf_gpu);
static float elapsed_time(struct timeval *start);
void printGPUDetails(FILE *fp);

        
/* global vars */
int nchan=128;
int ninp=4;
int debug=0;
int naver=99999;
int fft_batchsize=50;
char *infilename=NULL,*outfilename=NULL;
int2 inp_index[MAX_CORR_PROD];

int main(int argc, char * const argv[]) {
    size_t siz_inp_buf=0,siz_ft_buf=0,siz_corr_buf=0,siz_inp_gpu=0;
    int filedone=0,ncorr,iter=0,nav_written=0,i,j,cindex=0,nmoves=0,log2_2nchan;
    FILE *finp=NULL,*fout=NULL;
    char optstring[]="dc:i:o:n:a:f:";
    unsigned char *staging_buf=NULL,*staging_buf_gpu=NULL;
    cufftComplex *inp_buf_gpu=NULL;
    cufftComplex *ft_buf=NULL;
    cuFloatComplex *corr_buf=NULL;
    cuFloatComplex *corr_buf_gpu=NULL;
    cudaError_t res;
    int2 *inp_index_gpu=NULL;
    struct timeval thetime;
    float read_time=0,fft_time=0,cmac_time=0,sync_time=0,write_time=0,move_time=0,unpack_time=0;
    
    /* process command line args */
    if (argc <2) print_usage(argv);
    parse_cmdline(argc,argv,optstring);
    ncorr = ninp*(ninp+1)/2;
    
    /* calculate the size of various buffers in bytes */
    siz_inp_buf  = ninp*(nchan*2)*sizeof(unsigned char)*fft_batchsize;
    siz_inp_gpu  = ninp*(nchan*2)*sizeof(cufftComplex)*fft_batchsize;
    siz_ft_buf   = ninp*(nchan*2)*sizeof(cufftComplex)*fft_batchsize;
    siz_corr_buf = ncorr*(nchan)*sizeof(cufftComplex);

    if (debug) {
        printGPUDetails(stderr);
        fprintf(stderr,"---\nNum inp:\t%d. Num corr products: %d\n",ninp,ncorr);
        fprintf(stderr,"Num chan:\t%d\n",nchan);
        fprintf(stderr,"infile: \t%s\n",infilename);
        fprintf(stderr,"outfile:\t%s\n",outfilename);
        fprintf(stderr,"batchsize:\t%d\n",fft_batchsize);
        fprintf(stderr,"Input buffer size:\t%d\n",(int)siz_inp_buf);
        fprintf(stderr,"Size of uchar: %d\n",(int)sizeof(unsigned char));
    }

    /* check that nchan is power of 2 */
    for(i=1;i<30;i++) {
        // see if we can match 2*nchan with a power of 2
        if(1<<i == 2*nchan) {
            log2_2nchan=i;
            break;
        }
    }
    if (log2_2nchan ==0) {
        fprintf(stderr,"ERROR: number of channels must be a power of 2\n");
    }

    
    /* open input and output files */
    openFiles(infilename,outfilename,&finp,&fout);

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

    /* set to zero, since stuff is accumulated into these arrays */
    cudaMemset(corr_buf_gpu,'\0',siz_corr_buf);
    cudaMemset(ft_buf,'\0',siz_ft_buf);
    cudaMemset(inp_buf_gpu,'\0',siz_inp_gpu);

    /* calculate the execution configuration for the CMAC on the GPU */
    /* the CMAC is executed as a set of 1D threads. */
    dim3 dimBlock(nchan+1);
    dim3 dimGrid(ncorr);

    if ( ((2*nchan*ninp*fft_batchsize) % MAX_THREADS) != 0) {
        fprintf(stderr,"can't find integer mult of threads for unpack\n");
        exit(1);
    }

    //dim3 dimBlockUnpack(MAX_THREADS);
    //dim3 dimGridUnpack(2*nchan*ninp*fft_batchsize/MAX_THREADS);
    dim3 dimBlockUnpack(nchan);
    dim3 dimGridUnpack(2*ninp*fft_batchsize);

    /* make a lookup cache for which inputs went into a correlation product */
    for(i=0; i<ninp; i++) {
        for (j=i; j<ninp; j++){
            inp_index[cindex] = make_int2(i,j);
            cindex++;
        }
    }

    /* input index lookup. lives on GPU only */
    res = cudaMalloc((void **)&inp_index_gpu,sizeof(inp_index));
    if (res != 0) {
        fprintf(stderr,"failed to alloc device mem for inp_index_gpu\n");
        exit(1);
    }
    res = cudaMemcpy(inp_index_gpu,inp_index,sizeof(inp_index),cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
        fprintf(stderr,"Error on memcpy of inp_index. Message: %s\n",cudaGetErrorString(res));
        goto EXIT;
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
            unpack_data_GPU<<<dimGridUnpack,dimBlockUnpack>>>(nchan,ninp,fft_batchsize,log2_2nchan,staging_buf_gpu,inp_buf_gpu);
            cudaDeviceSynchronize(); /* must wait before starting FFT */
            unpack_time += elapsed_time(&thetime);            
            if ( (res=cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr,"unpack_data_GPU failed. Error: %s\n",cudaGetErrorString(res));
                goto EXIT;
            }

            /* do the FFT */
            gettimeofday(&thetime,NULL);
            if (do_FFT_gpu(nchan,ninp,inp_buf_gpu,ft_buf) != CUFFT_SUCCESS) goto EXIT;
            fft_time += elapsed_time(&thetime);

            /* do the CMAC. don't sync after this call since the next batch of data can be read in parallel. */
            gettimeofday(&thetime,NULL);
            do_CMAC_gpu<<<dimGrid,dimBlock>>>(nchan,ninp,ncorr,fft_batchsize,inp_index_gpu,
                (cuFloatComplex *)ft_buf,(cuFloatComplex *) corr_buf_gpu);
            cmac_time += elapsed_time(&thetime);
        }
        
        /* write out if it is time to */
        if (filedone || (++iter)*fft_batchsize >= naver) {
            cudaDeviceSynchronize();
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
             
            writeOutput(fout,ncorr,nchan,1.0/(nchan*iter*fft_batchsize),corr_buf);
            if(debug) fprintf(stderr,"writing average of %d chunks\n",iter*fft_batchsize);
            iter=0;
            nav_written++;
            write_time += elapsed_time(&thetime);
        }
    }
    
    if (debug) {
        fprintf(stderr,"wrote %d averages. unused chunks: %d\n",nav_written,iter*fft_batchsize);
        fprintf(stderr,"Time reading:\t%g ms\n",read_time);
        fprintf(stderr,"Time moving:\t%g ms. N moves: %d, BW: %g GB/s\n",
                move_time,nmoves,(float)nmoves*(float)siz_inp_buf/(move_time*1e-3)*1e-9);
        fprintf(stderr,"Time unpacking:\t%g ms\n",unpack_time);
        fprintf(stderr,"Time FFTing:\t%g ms\n",fft_time);
        fprintf(stderr,"Time CMACing:\t%g ms\n",cmac_time);
        fprintf(stderr,"Time syncing:\t%g ms\n",sync_time);
        fprintf(stderr,"Time writing:\t%g ms\n",write_time);
    }

EXIT:
    /* clean up */
    fclose(finp);
    fclose(fout);
    if (staging_buf_gpu != NULL) cudaFree(staging_buf_gpu);
    if (staging_buf != NULL) cudaFreeHost(staging_buf);
    if (ft_buf != NULL) cudaFree(ft_buf);
    if (inp_buf_gpu != NULL) cudaFree(inp_buf_gpu);
    if (corr_buf != NULL) cudaFreeHost(corr_buf);
    if (corr_buf_gpu != NULL) cudaFree(corr_buf_gpu);
    if (inp_index_gpu != NULL) cudaFree(inp_index_gpu);
    return 0;
}


/* unpack the byte data into float format on the GPU.
   The input is a sequence of bytes, one for each input for each time instant. input i, time t
   i.e. i0t0,i1t0,i2t0,i3t0,i0t1,i1t1,i2t1,i3t1,i0t2, etc.
   We need to re-order these into
   sequences for inputs in the FFT. i.e.
   i0t0,i0t1,...i0,t(2*nchan-1),i1,t0,i1,t1,...i1,t(2*nchan-1)

   The data is unpacked into complex floats so that the FFT buffer alignment is correct for the results
   of the FFT. See below.
*/
__global__ void unpack_data_GPU(const int nchan,const int ninp,const int batchsize, const int log2_2nchan,
                                unsigned char *staging_buf, cufftComplex *inp_buf) {
    int output_index,batch_index,intra_batch_index,input_index,time_index,sample_index;
    const int two_nchan = 2*nchan;
    
    /* calculate the index in the destination array */
    output_index = blockIdx.x*nchan + threadIdx.x;
    
    /* as per CUDA manual, integer division and modulo are very expensive, so use bitwise operations instead
       which are valid for power-of-2 divisors if possible */
    /* we can probably avoid all this by setting up a lookup table array at the beginning */
    batch_index = output_index/(two_nchan*ninp);
    //intra_batch_index = output_index%(two_nchan*ninp);
    intra_batch_index = output_index&(two_nchan*ninp -1);
    input_index = intra_batch_index >> log2_2nchan;
    time_index  = intra_batch_index&(two_nchan -1);
    sample_index= input_index + ninp*(time_index + batch_index*two_nchan);
#ifdef __DEVICE_EMULATION__
    /*printf("blk: %d, thr: %d, outind: %d, batind: %d, ibi: %d, inpind: %d, timind: %d, sampind: %d\n",
                blockIdx.x,threadIdx.x,output_index,batch_index,intra_batch_index,input_index,time_index,sample_index);
     */
#endif
    //inp_buf[output_index] = make_cuFloatComplex((int)staging_buf[sample_index] - 128.0,0.0);
    inp_buf[output_index] = make_cuFloatComplex((int)staging_buf[sample_index],0.0);
}


 
/*  Do the CMAC on the GPU. At this stage, we have a contiguous buffer of dim [fft_batchsize][ninp][2*nchan]
    complex numbers in ft_buf. The FFT has been done as complex->complex, so the second half of the 2*chan
    array is redundant.
    We need to accumulate these into correlation products in corr_buf. nchan+1 threads are executed, each
    of them reading consecutive memory locations. This way we get all channels in the FFTd spectrum including
    the DC term. The first thread (==0) with the DC term is discarded for reasons explained below.
*/
/*  this function is where most of the action happens. We want to guarantee coalesced memory
    reads across threads. This means aligning the arrays correctly. For this reason, we choose
    to drop the DC component of the spectrum at this point in processing. The output array is then
    just nchan*ncorr, so the beginning of each nchan sub-array correlation product will always be aligned
    for power-of-two FFT sizes. This guarantees colasced writes to the output array.
    To guarantee coalesced reads, the first (DC) component of FFTd spectrum is read. So, a total of nchan+1
    threads are required. The zero'th thread just doesn't write anything out and the (n==1) thread writes
    to output array index 0 and so on. This shifts the results of cmac down by one, discarding the DC component and
    writing out nchan results, not nchan+1.
*/
__global__ void do_CMAC_gpu(const int nchan, const int ninp, const int ncorr,const int batchsize,
        const int2 *inp_index_gpu, cuFloatComplex *ft_buf, cuFloatComplex *corr_buf) {

    /* block index */
    int blk_ind = blockIdx.x;
    /* thread index. */
    int thr_ind = threadIdx.x;
    
    /* define inputs (a,b) and result (c) */
    cuFloatComplex a,b,c;
    int cind=0,batch_ind=0,inp1_offset,inp2_offset;
    int2 inp_index;

    cind = blk_ind%(ncorr);

    /* fetch the pre-calculated input indexes for this correlation product */
    inp_index = inp_index_gpu[cind];

#ifdef __DEVICE_EMULATION__
#endif

    c = make_cuFloatComplex(0.0,0.0);

    /* loop over all the time indices (one for each FFT batch) since they all go into the same product */
    for(batch_ind=0; batch_ind<batchsize; batch_ind++) {

        inp1_offset = (batch_ind*ninp+inp_index.x)*2*nchan;
        a = ft_buf[inp1_offset + thr_ind];

        if (inp_index.x==inp_index.y) {
            /* don't re-fetch for autocorrelations */
            inp2_offset = inp1_offset;
            b = a;
        } else {
            inp2_offset = (batch_ind*ninp+inp_index.y)*2*nchan;
            b = ft_buf[inp2_offset + thr_ind];
        }
#ifdef __DEVICE_EMULATION__
        if (batch_ind==0) printf("blk: %d, thr: %d, cind: %d, batch: %d, inp1_offset: %d, inp2_offset: %d\n",
           blk_ind,thr_ind,cind,batch_ind,inp1_offset,inp2_offset);
#endif
        /* do the complex mult (including conjugation) */
        c = cuCaddf(c,cuCmulf(a,cuConjf(b)));

    }
    
    //__syncthreads();

    /* accumulate and store result */    
    if (thr_ind != 0) {
        corr_buf[cind*nchan + thr_ind -1] = cuCaddf(corr_buf[cind*nchan + thr_ind-1],c);
    }
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
       and ninp*(nchan+1)*sizeof(complex) for result */
    if (!doneplan) {
        
        res = cufftPlan1d(&plan,nchan*2,CUFFT_C2C,ninp*fft_batchsize);
        if (res != CUFFT_SUCCESS) {
            fprintf(stderr,"ERROR: cufftPlan1d failed with error code %d\n",res);
            return res;
        }
        if (debug) fprintf(stderr,"Made a plan (id: %d) for %d channel FFT with batch size %d*%d\n",
                        (int) plan,nchan*2,ninp,fft_batchsize);
        doneplan=1;
    }
    
    res = cufftExecC2C(plan,inp_buf,ft_buf,CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {
        fprintf(stderr,"cuda FFT failed with result code %d\n",res);
        return res;
    }
    return 0;
}


/* write out correlation products. Ignore the first (DC) component */
/* note that nchan does not include the DC component */
void writeOutput(FILE *fout,int ncorr, int nchan, const float normaliser, cuFloatComplex *buf) {
    int cprod,chan;
    cuFloatComplex cNorm;
    
    cNorm = make_cuFloatComplex(normaliser,0.0);
        
    for (cprod=0; cprod<ncorr; cprod++) {
        for(chan=0; chan<nchan; chan++) {
            buf[cprod*nchan+chan] = cuCmulf(buf[cprod*nchan+chan],cNorm);
            //printf("cprod: %d, chan: %d, val: %g,%g\n",cprod, chan, buf[cprod*nchan+chan][0],buf[cprod*nchan+chan][1]);
        }
        fwrite(buf+cprod*nchan,sizeof(cuFloatComplex),nchan,fout);
        //fflush(fout);
    } 
}


/* incoming data is a stream of bytes, one per input per time sample. This needs to be packed into arrays
   where 2*nchan values for the same input channel are contiguous. Also, for FFT batching, we load fft_batchsize
   time chunks (1 chunk = 2*nchan values per channel) at a time. This then means we have fft_batchsize sets of
   ninp sets of 2*nchan samples per call to this function */
int readData(int nchan,int ninp,FILE *fpin,unsigned char *inp_buf) {
    int ntoread=0;
    
    int nread;

    ntoread = ninp*sizeof(unsigned char)*nchan*2*fft_batchsize;
    
    nread = fread(inp_buf,sizeof(unsigned char),ninp*nchan*2*fft_batchsize,fpin);
    if(nread < ntoread) return 1;
            
    return 0;
}


int openFiles(char *infilename, char *outfilename, FILE **fin, FILE **fout) {
    
    if(infilename ==NULL || outfilename==NULL) {
        fprintf(stderr,"ERROR: NULL filenames\n");
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
    
    if (strcmp(outfilename,"-")==0) {
        *fout = stdout;
    } else {
        *fout = fopen(outfilename,"w");
        if (*fout ==NULL) {
            fprintf(stderr,"failed to open output file name: <%s>\n",outfilename);
            exit(1);
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
                if (naver <=0 || naver > 65536) {
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
            case 'd':
                debug++;
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
        fprintf(fp,"MEM:\t\t%d MB\n",(int)((devprop.totalGlobalMem)/(1024*1024)));
        fprintf(fp,"Shmem per block:\t%d\n",(int)devprop.sharedMemPerBlock);
        fprintf(fp,"regs  per block:\t%d\n",devprop.regsPerBlock);
        fprintf(fp,"Warp size:\t%d\n",devprop.warpSize);
        fprintf(fp,"Mem pitch:\t%d\n",(int)devprop.memPitch);
        fprintf(fp,"Max thr per blk:\t%d\n",devprop.maxThreadsPerBlock);
        fprintf(fp,"Max dim per blk:\t%d,%d,%d\n",(int)devprop.maxThreadsDim[0],(int)devprop.maxThreadsDim[1],(int)devprop.maxThreadsDim[2]);
        fprintf(fp,"Tot const mem:  \t%d\n",(int)devprop.totalConstMem);
        fprintf(fp,"Version:\t%d.%d\n",(int)devprop.major,(int)devprop.minor);
        fprintf(fp,"clockrate:\t%d\n",(int)devprop.clockRate);
        fprintf(fp,"texture algn:\t%d\n",(int)devprop.textureAlignment);
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels. default: %d\n",nchan);
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. default: %d\n",ninp);
    fprintf(stderr,"\t-a num\t\tspecify number of averages before output. default: %d\n",naver);
    fprintf(stderr,"\t-f num\t\tspecify fft batchsize. default: %d\n",fft_batchsize);
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");
    fprintf(stderr,"\t-d    \t\tenable debugging output\n");
    exit(0);
}
