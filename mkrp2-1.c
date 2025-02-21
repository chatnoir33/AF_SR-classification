#include <stdio.h>
#include <math.h>
#define NN 1000 // #data

int main(int argc, char *argv[]) {
    int i, j, k, l, iter=0, eof=0;
    double x[NN];
    double zmax=0.0, eps=0.0, ave=0.0;
	double tmp, temp;
	int count=0,count0=0, count1=0;
	double sign, norm1, norm2;
	int num, n1;
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
	dim = 1;
	delay = 1;
    fscanf(fp1,"%d",&num);
    fclose(fp1);
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
            for(j=1;j<=n1;j++){
                // CalcDistance
				temp = 0;
            	for(k=0;k<dim;k++){
                	temp+=((x[i+(k)*delay]-x[i+j+(k)*delay])*(x[i+(k)*delay]-x[i+j+delay*(k)]));
				}
				temp=sqrt(temp);
                if(temp>zmax) zmax=temp;
            }
        }
        printf("P2\n%d %d\n256\n",n1,n1);
        for(i=0;i<n1;i++){
            for(j=1;j<=n1;j++){
				ave = 0;
            	for(k=0;k<dim;k++){
                	ave+=((x[i+(k)*delay]-x[i+j+(k)*delay])*(x[i+(k)*delay]-x[i+j+delay*(k)]));
				}
				ave=sqrt(ave)/zmax;
				eps += ave;
				count++;
            }
        }
		eps = eps/(count);
        for(i=0;i<n1;i++){
            for(j=1;j<=n1;j++){
				ave = 0;
				norm1 = 0;
				norm2 = 0;
            	for(k=0;k<dim;k++){
                	ave+=((x[i+(k)*delay]-x[i+j+(k)*delay])*(x[i+(k)*delay]-x[i+j+delay*(k)]));
					norm1 += x[i+(k)*delay]*x[i+(k)*delay];
					norm2 += x[i+j+(k)*delay]*x[i+j+(k)*delay];
				}
				norm1 = sqrt(norm1);
				norm2 = sqrt(norm2);
				sign = norm1-norm2;
				ave=sqrt(ave)/zmax;
				if(ave < eps){
					if(sign > 0){
						printf("255\n");
					}
					else{
						printf("128\n");
					}
					count0++;
				}
				else{
					if(sign > 0){
						printf("64\n");
					}
					else{
						printf("0\n");
					}
					count1++;
				}
            }
        }
    return 0;
}
