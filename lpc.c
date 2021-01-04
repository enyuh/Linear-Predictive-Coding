#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "mtx.h"

#define sRate 16000
#define PI 3.14159265359
#define ANALYSIS_WINDOW_SIZE_IN_SMP 512
#define P_TH_ORDER 20
#define LOW_PITCH 60
#define HIGH_PITCH 300

float *hamwnd;
void HammingWndCon(size_t len)
{
	size_t i;
	float radian;
	hamwnd = (float *) calloc(sizeof(float), len) ;
	for(i=0;i<len;i++) {
		radian = 2.0 * PI * ((float) i) / ((float) (len-1) );
		hamwnd[i] = 0.54 - 0.46 * cos( radian );
	}
}

void HammingWndDes(void)
{
	if( hamwnd )
		free( hamwnd );
}

double crosscorr(int N, double *x, int k, int i)
{
	int j;
	double ret = 0;
	int k_tmp, i_tmp;
	for(j=0;j<N;j++) {
		k_tmp = j - k;
		i_tmp = j - i;
		if( k_tmp>=0 && i_tmp>=0 ){
			ret += x[k_tmp] * x[i_tmp];
		}
	}
	return ret;
}



void lpc(int N, double *x, int P, double *a)
{
	double **A = NULL;
	double **ak = NULL;
	double **B = NULL;
	double **invA = NULL;
	int k, i;
	mtxalc(&A, P, P);
	mtxalc(&invA, P, P);
	mtxalc(&ak, P, 1);
	mtxalc(&B, P, 1);
	for(i=1;i<=P;i++) {
		for(k=1;k<=P;k++) {
			A[i-1][k-1] = crosscorr(N, x, k, i);
		}
		B[i-1][0] = crosscorr(N, x, 0, i);
	}

	mtxinv(A, invA, P);
	mtxmtp(ak, invA, B, P, P, P, 1);

	for(i=1;i<=P;i++) {
		a[i] = ak[i-1][0];
	}

	mtxfree(A, P, P);
	mtxfree(invA, P, P);
	mtxfree(ak, P, 1);
	mtxfree(B, P, 1);
}

void lpc_residual(int N, double *x, double *pre_x, double *ak, int P, double *residual)
{
	int n, k;
	double x_hat;

	for(n=0;n<P;n++) {	
		x_hat = 0;
		for(k=1;k<=P;k++) {
			if( (n-k) < 0 ) {
				x_hat += ak[k] * pre_x[N+n-k];
			}
			else {
				x_hat += ak[k] * x[n-k];
			}
		}
		residual[n] = x[n] - x_hat;
	}

	for(n=P;n<N;n++) {	
		x_hat = 0;
		for(k=1;k<=P;k++) {
			x_hat += ak[k] * x[n-k];
		}
		residual[n] = x[n] - x_hat;
	}
}


void lpc_synthesis(int N, double *residual, double *pre_x, double *ak, int P, double *x)
{
	int n;
	int k;
	double ini_state_value[100];
	double predicted_x;
	for(n=1;n<=P;n++) {
		ini_state_value[n] = pre_x[N-n];
	}
	for(n=0;n<N;n++) {
		predicted_x = 0;
		for(k=1;k<=P;k++) {
			predicted_x += ak[k] * ini_state_value[k];
		}
		x[n] = predicted_x + residual[n];

		// delay shift for next for loop (for (n+1) )
#ifdef VERY_SILLY_ERROR
		for(k=1;k<P;k++) {
			ini_state_value[k+1] = ini_state_value[k];
		}
#else
		for(k=P-1;k>0;k--) {
			ini_state_value[k+1] = ini_state_value[k];
		}
#endif
		if( x[n]>32768 || x[n]<(-32768)) {
			//x[n] = 0;
		}

		ini_state_value[1] = x[n];
		
	}
}

int simple_f0_extractor(int N, double *x, float th)
{
	int n;
	int ret = 0;
	double tmp;
	double low_pitch_lag = 1.0 / ((double)LOW_PITCH) * sRate;
	double high_pitch_lag = 1.0 / ((double)HIGH_PITCH) * sRate;
	double eng = crosscorr(N, x, 0, 0);
	double max_c = crosscorr(N, x, high_pitch_lag, 0) / sqrt(crosscorr(N, x, 0, 0)) / sqrt(crosscorr(N, x, high_pitch_lag, high_pitch_lag));

	if( high_pitch_lag < 1 ) high_pitch_lag = 1;
	if( low_pitch_lag >= 1 ) low_pitch_lag = N-1;
	ret = 1600000;
	fprintf(stdout, "%1.3lf ", max_c);
	for(n=((int)high_pitch_lag);n<((int)low_pitch_lag);n++) {
		tmp = crosscorr(N, x, n, 0) / sqrt(crosscorr(N, x, 0, 0)) / sqrt(crosscorr(N, x, n, n));
		fprintf(stdout, "%1.3lf ", tmp);
		if( tmp > max_c ) {
			max_c = tmp;
			ret = n;
		}
	}
	if( max_c > th ) {
		ret = ret;
	}
	else {
		ret = 16000;
	}

	fprintf(stdout, "\n");
	ret = ret;
}

int main(void)
{
	char pcmfn[256] = {"input.wav"};
	size_t total_len_in_byte = 0;
	size_t pcm_len_in_byte = 0;
	size_t pcm_len_in_smp = 0;
	FILE *fp = fopen(pcmfn, "rb");
	FILE *fp_save = NULL;
	FILE *fp_save_residual = fopen("residual.pcm", "wb");
	FILE *fp_save_residual_normalized = fopen("residual_normalized.pcm", "wb");
	FILE *fp_save_residual_artificial = fopen("residual_artificial.pcm", "wb");
	FILE *fp_save_gain = fopen("gain.pcm", "wb");
	FILE *fp_save_artificial_pitch = fopen("artificial_pitch.pcm", "wb");
	FILE *fp_load_lpc = NULL;
	FILE *fp_load_residual = NULL;
	FILE *fp_save_reconstructed = NULL;
	FILE *fp_save_reconstructed_ascii = NULL;
	FILE *fp_save_pitch = fopen("pitch.txt", "w");
	FILE *fp_load_pitch;
	short *pcmbuff = NULL;
	float *xm;
	float *magnitude;
	double *data_ori;
	double *data_ori_pre;
	double *residual;
	float *fresidual;
	float *artificial_residual;
	double ak[100] = {0};
	float fdata;
	int cplen;
	size_t n = 0;
	int N = 0;
	size_t m;
	float *x_re = NULL,  *x_im = NULL, *X_re = NULL, *X_im = NULL, *C = NULL, *ReSpec = NULL;
	double *data;
	int frm_num;
	size_t k;
	size_t window_size_in_ms = 25, frame_size_in_ms = 5;
	size_t window_size_in_smp = sRate * window_size_in_ms / 1000;
	size_t frame_size_in_smp =  sRate * frame_size_in_ms / 1000;
	int left_b, right_b;
	int P = P_TH_ORDER;
	float period;
	float pitch;
	float residual_eng;
	float gain;
	float frame_pitch[10000];
	int smp_ofs;


	fseek(fp, 0, SEEK_END);
	total_len_in_byte = ftell(fp);
	pcm_len_in_byte = total_len_in_byte - 44;
	pcm_len_in_smp = pcm_len_in_byte / 2;
	fseek(fp, 44, SEEK_SET);
	pcmbuff = (short *) calloc(sizeof(short), pcm_len_in_smp);
	artificial_residual = (float *) calloc(sizeof(float), pcm_len_in_smp);
	x_re = (float *) calloc(sizeof(float), pcm_len_in_smp);
	x_im = (float *) calloc(sizeof(float), pcm_len_in_smp);
	xm = (float *) calloc(sizeof(float), ANALYSIS_WINDOW_SIZE_IN_SMP);
	magnitude = (float *) calloc(sizeof(float), ANALYSIS_WINDOW_SIZE_IN_SMP);
	data_ori = (double *) calloc(sizeof(double), frame_size_in_smp);
	data_ori_pre = (double *) calloc(sizeof(double), frame_size_in_smp);
	residual = (double *) calloc(sizeof(double), frame_size_in_smp);
	fresidual = (float *) calloc(sizeof(float), frame_size_in_smp);
	data = (double *) calloc(sizeof(double), window_size_in_smp);
	HammingWndCon(window_size_in_smp);
	fread(pcmbuff, sizeof(short), pcm_len_in_smp, fp);
	for(n=0;n<pcm_len_in_smp;n++) {
		x_re[n] = (float)(pcmbuff[n]);
	}

	fclose(fp);

	n = frame_size_in_smp/2;// offset of frame centroid
	frm_num = 0;
	fp_save = fopen("lpc.dat", "wb");
	while( n < pcm_len_in_smp ) {
		frm_num ++;
		memset(xm, 0, sizeof(float) * ANALYSIS_WINDOW_SIZE_IN_SMP);
		if( (n+window_size_in_smp)>=pcm_len_in_smp ) {
			cplen = pcm_len_in_smp - n;
		}
		else {
			cplen = window_size_in_smp;
		}
		left_b = n - window_size_in_smp / 2;
		right_b = n + window_size_in_smp / 2;
		if( left_b < 0 ) {
			if( right_b >= pcm_len_in_smp ) {
				memcpy(xm - left_b, x_re, sizeof(float) * pcm_len_in_smp);
			}
			else {
				memcpy(xm - left_b, x_re, sizeof(float) * right_b);
			}
		}
		else {
			if( right_b >= pcm_len_in_smp ) {
				memcpy(xm, x_re + left_b, sizeof(float) * (pcm_len_in_smp-left_b));
			}
			else {
				memcpy(xm, x_re + left_b, sizeof(float) * (right_b-left_b));
			}
		}
		
		// apply hamming window
		for(m = 0; m < window_size_in_smp; m ++ ) {
			xm[m] = xm[m] * hamwnd[m];
			data[m] = (double) (xm[m]);// data for LPC analysis
		}

		// copy the data for finding residual
		for(m = 0; m < frame_size_in_smp; m ++ ) {
			k = n - frame_size_in_smp/2 + m;
			if( k<0 || k>=pcm_len_in_smp ) {
				data_ori[m] = 0;
			}
			else {
				data_ori[m] = x_re[k];
			}			
		}

		// extract LPC coefficients
		lpc(window_size_in_smp, data, P, ak);
		for(m = 0; m < (P+1); m ++ ) {
			fwrite(&(ak[m]), sizeof(double), 1, fp_save);
			//fprintf(fp_save, "%.15e\t", ak[m]);
		}

		// extract pitch
		period = (float)simple_f0_extractor(window_size_in_smp, data, 0.6);
		pitch = 1 / (period / sRate);
		if( pitch < LOW_PITCH || pitch > HIGH_PITCH ) {
			pitch = 0;
		}
		fprintf(fp_save_pitch, "%lf\n", pitch);
		//fprintf(fp_save, "\n");

		// find residual signal e[n]
		lpc_residual(frame_size_in_smp, data_ori, data_ori_pre, ak, P, residual);
		for(m = 0; m < frame_size_in_smp; m ++ ) {
			//fprintf(fp_save_residual, "%lf\n", residual[m]);
			fdata = (float)residual[m];
			fwrite(&fdata, sizeof(float), 1, fp_save_residual);
		}

		// find gain
		residual_eng = 0;
		for(m = 0; m < frame_size_in_smp; m ++ ) {
			residual_eng += (float)residual[m] * (float)residual[m];
		}
		gain = sqrt(residual_eng);
		fwrite(&gain, sizeof(float), 1, fp_save_gain);

		// find normalized residual
		for(m = 0; m < frame_size_in_smp; m ++ ) {
			fdata = (float)residual[m];
			fdata /= gain;
			fwrite(&fdata, sizeof(float), 1, fp_save_residual_normalized);
		}

		n += frame_size_in_smp;
		memcpy(data_ori_pre, data_ori, sizeof(double)*frame_size_in_smp);
	}
	fclose(fp_save);

	HammingWndDes();

	fclose(fp_save_residual);

	free(pcmbuff);

	fclose(fp_save_pitch);

	fclose(fp_save_residual_normalized);
	fclose(fp_save_gain);
	fclose(fp_save_artificial_pitch);


	/* generate artificial pitch */
	fp_load_pitch = fopen("pitch.txt", "r");
	for(m=0;m<frm_num;m++) {
		fscanf(fp_load_pitch, "%f", &(frame_pitch[m]));
	}
	fclose(fp_load_pitch);
	// generate white noise of constant energy
	for(m=0;m<pcm_len_in_smp;m++) {
		artificial_residual[m] = (rand() % 2) * 2 - 1;
	}
	for(m=0;m<frm_num;m++) {
		frame_pitch[m] = frame_pitch[m] * 2.0;
	}
	m = 0;
	smp_ofs = 0;
	while( smp_ofs < pcm_len_in_smp ) {
		if( frame_pitch[smp_ofs/frame_size_in_smp] != 0 ) {
		//	frame_pitch[smp_ofs/frame_size_in_smp];
			artificial_residual[smp_ofs] = ((1.0/frame_pitch[smp_ofs/frame_size_in_smp])*sRate);
			smp_ofs += ((int)artificial_residual[smp_ofs]);
		}
		else {
			smp_ofs += frame_size_in_smp;
		}		
	}
	fwrite(artificial_residual, sizeof(float), pcm_len_in_smp, fp_save_residual_artificial);
	fclose(fp_save_residual_artificial);

	/*----- reconstruct the speech from residuals and lpc coefficients -----*/
	fp_load_lpc = fopen("lpc.dat", "rb");
	//fp_load_residual = fopen("residual.pcm", "rb");
	fp_load_residual = fopen("residual_artificial.pcm", "rb");
	fp_save_reconstructed = fopen("reconstructed.pcm", "wb");
	fp_save_reconstructed_ascii = fopen("reconstructed.txt", "w");
	n = frame_size_in_smp/2;// offset of frame centroid
	memset(data_ori_pre, 0, sizeof(double) * frame_size_in_smp);// setting initial-rest condition
	for(m=0;m<frm_num;m++) {
		// load lpc coeffients for each frame
		for(k=0;k<=P;k++) {
			fread(&(ak[k]), sizeof(double), 1, fp_load_lpc);
		}
		memset(data_ori, 0, sizeof(double) * frame_size_in_smp);
		memset(residual, 0, sizeof(double) * frame_size_in_smp);
		if( (n+frame_size_in_smp)>=pcm_len_in_smp ) {
			cplen = pcm_len_in_smp - n;
		}
		else {
			cplen = frame_size_in_smp;
		}
		left_b = n - window_size_in_smp / 2;
		right_b = n + window_size_in_smp / 2;
		fread(fresidual, sizeof(float), cplen, fp_load_residual);

		for(k=0;k<frame_size_in_smp;k++) {
			residual[k] = (double) fresidual[k];
		}
		lpc_synthesis(frame_size_in_smp, residual, data_ori_pre, ak, P, data_ori);
		for(k=0;k<frame_size_in_smp;k++) {
			fdata= (float) data_ori[k];
			fwrite(&fdata, sizeof(float), 1, fp_save_reconstructed);
			fprintf(fp_save_reconstructed_ascii, "%.15f\n", fdata);
		}
		
		// prepare the data for the next frame
		memcpy(data_ori_pre, data_ori, sizeof(double) * frame_size_in_smp);
		n += frame_size_in_smp;
	}


	fclose(fp_load_lpc);
	fclose(fp_load_residual);
	fclose(fp_save_reconstructed);
	fclose(fp_save_reconstructed_ascii);
	return 1;
}