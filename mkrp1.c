#include <stdio.h>
#include <math.h>
#define NN 1000 // #data

int main(int argc, char *argv[]) {
    int i, j, k, l, iter=0, eof=0;
    double x[NN]; // "t[N]"IsDummy
    double zmax=0.0, eps=0.0, ave=0.0;
	double tmp, temp;
	int count=0,count0=0, count1=0;
	int num, nn, n1;
	int dim, delay;
    FILE *fp1, *fp2;
    
    if(argc!=3) {
        printf("missing file argument\n");
        return 1;
    }
    if((fp1=fopen(argv[1],"r"))==NULL) {
        printf("can't open %s\n", argv[1]);
        return 1;
    }
	dim = 3;
	delay = 3;
    fscanf(fp1,"%d",&num);
    fclose(fp1);
	nn = num/2;
	n1 = 256;

    if((fp2=fopen(argv[2],"r"))==NULL) {
        printf("can't open %s\n", argv[1]);
        return 1;
    }
    
        for(i=0;i<num;i++){
            if(fscanf(fp2,"%lf",&x[i])!=1){
                eof=1;
				printf("data is less than %d\n",num);
				exit(1);
            }
        }
    	fclose(fp2);
        for(i=0;i<n1;i++){
            for(j=0;j<n1;j++){
                // CalcDistance
				temp = 0;
            	for(k=0;k<dim;k++){
                	temp+=(x[i+(k)*delay]-x[i+j+(k)*delay])*(x[i+(k)*delay]-x[i+j+delay*(k)]);
				}
				temp=sqrt(temp);
                if(temp>zmax) zmax=temp;
            }
        }
        printf("P1\n%d %d\n1\n",256,256);
		eps = 0.1;
        for(i=0;i<n1;i++){
            for(j=0;j<n1;j++){
				ave = 0;
            	for(k=0;k<dim;k++){
                	ave+=(x[i+(k)*delay]-x[i+j+(k)*delay])*(x[i+(k)*delay]-x[i+j+delay*(k)]);
				}
				ave=sqrt(ave)/zmax;
				if(ave < eps){
					printf("1\n");
					count1++;
				}
				else{
					printf("0\n");
					count0++;
				}
            }
        }
		printf(" %lf\n",(double)count1/(count1+count0));
    return 0;
}
