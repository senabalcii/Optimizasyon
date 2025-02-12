#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define MAX_ITER 1500


typedef struct {
    int sinif;
    float* vektor;
} goruntu;

float* readPGMImage(const char* filename, int* width, int* height);
float tanhHesapla(float* w, float* x, int size);
float basariOrani(goruntu* goruntuler, int size, float* w, int vectorsize);
void bolEgitimTestKumesi(goruntu* goruntuler, int numFiles,goruntu* egitimKumesi, goruntu* testKumesi);
float* gradyanHesaplama(goruntu* goruntuler, float* w, float* grad, int vectorsize, int trainsize);
float hesaplaKayip(goruntu* goruntuler, int size, float* w, int vectorsize);
float* gradientDescent(float* w, float* grad, int vectorSize, int trainSize, goruntu* goruntuler, float* trainLosses,float* basari);
float* gradyanHesaplaMiniBatch(goruntu* goruntuler, float* w, float* grad, int vectorsize, int trainsize, int batchSize);
float* stochasticGradientDescentMiniBatch(float* w, float* grad, int vectorSize, int trainSize, goruntu* goruntuler, float* trainLosses,float* basari);
float* adam(float* w, int vectorsize, int trainsize, float* grad, goruntu* goruntuler, float* trainLosses,float* basari);
void saveWeightsToFile(const char* filename, float* weights, int vectorSize, int iter);
void saveLossToFile(const char* filename, int* epochs, float* losses, int size);
void saveTestLossToFile(const char* filename, int* epochs, float* losses, int size);
void saveAccToFile(const char* filename, int* epochs, float* basari, int size);
void rastgeleAgirlikOlustur(float* w, int vectorSize);



float* readPGMImage(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Dosya açılamadı: %s\n", filename);
        return NULL;
    }

    char format[3];
    fscanf(file, "%s", format);
    if (strcmp(format, "P5") != 0) {
        printf("Geçersiz format: %s\n", filename);
        fclose(file);
        return NULL;
    }

    fscanf(file, "%d %d", width, height);
    int maxval;
    fscanf(file, "%d", &maxval);
    fgetc(file); // Bir boşluk atla

    int size = (*width) * (*height);
    float* goruntu_vektoru = (float*)malloc((size + 1) * sizeof(float));
    unsigned char* buffer = (unsigned char*)malloc(size * sizeof(unsigned char));

    fread(buffer, sizeof(unsigned char), size, file);
	int i;
    for (i = 0; i < size; i++) {
        goruntu_vektoru[i] = buffer[i] / 255.0;
    }
    goruntu_vektoru[size] = 1.0; // Bias terimi

    free(buffer);
    fclose(file);
    return goruntu_vektoru;
}


float tanhHesapla(float* w, float* x, int size) {
    if (w == NULL || x == NULL) {
        printf("Hata: w veya x NULL pointer!\n");
        return 0.0;
    }

    if (size <= 0) {
        printf("Hata: Geçersiz boyut: %d\n", size);
        return 0.0;
    }

    float sonuc = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        sonuc += w[i] * x[i];
    }
    return tanh(sonuc);
}

float basariOrani(goruntu* goruntuler, int size, float* w, int vectorsize) {
    int dogruSayisi = 0;
    int i;
    float ypred;
    for (i = 0; i < size; i++) {

        ypred = tanhHesapla(w, goruntuler[i].vektor, vectorsize);
        if(ypred>0){
            ypred=1;
        }
        else{
            ypred=-1;
        }
        if (ypred == goruntuler[i].sinif) {
            dogruSayisi++;
        }
    }
    float basari = (float)dogruSayisi / size * 100;
    printf("Başarı Oranı: %% %.2f\n", basari);
    return basari;
}


void bolEgitimTestKumesi(goruntu* goruntuler, int numFiles,goruntu* egitimKumesi, goruntu* testKumesi) {
    int testSayisi = numFiles / 5; 
    int egitimSayisi = numFiles - testSayisi;

    int* indis = (int*)malloc(numFiles * sizeof(int));
    int i,j;
    for (i = 0; i < numFiles; i++) indis[i] = i;
    for (i = 0; i < numFiles; i++) {
        int j = rand() % numFiles;
        int temp = indis[i];
        indis[i] = indis[j];
        indis[j] = temp;
    }

    for (i=0; i<testSayisi; i++) {
        testKumesi[i]=goruntuler[indis[i]];
    }
    
    
    for (i=0; i<egitimSayisi; i++) {
        egitimKumesi[i]=goruntuler[indis[i+testSayisi]];
    }
    

    free(indis);
}

float* gradyanHesaplama(goruntu* goruntuler, float* w, float* grad, int vectorsize, int trainsize) {
    memset(grad, 0, vectorsize * sizeof(float));
    int i,j;
    for (i = 0; i < trainsize; i++) {
        float yTrue = goruntuler[i].sinif;
        float ypred = tanhHesapla(w, goruntuler[i].vektor, vectorsize);
        float sayi = (ypred - yTrue) * (1 - ypred * ypred);
        for (j = 0; j < vectorsize; j++) {
            grad[j] += goruntuler[i].vektor[j] * sayi;
        }
    }
    for (j = 0; j < vectorsize; j++) {
        grad[j] = (grad[j]*2)/trainsize;
    }
    return grad;
}

float hesaplaKayip(goruntu* goruntuler, int size, float* w, int vectorsize) {
    float toplamKayip = 0.0;
    int i;
	for (i = 0; i < size; i++) {
        float ypred = tanhHesapla(w, goruntuler[i].vektor, vectorsize);
        float yTrue = goruntuler[i].sinif;
        toplamKayip += (ypred - yTrue) * (ypred - yTrue);
    }
    return toplamKayip / size;
}



float* gradientDescent(float* w, float* grad, int vectorSize, int trainSize, goruntu* goruntuler, float* trainLosses,float* basari) {
    float eps = 0.01;           
    float precision = 1e-5;    
    float previousLoss = 0.0;  
    float currentLoss = 0.0;   
     
    FILE* file = fopen("weights_GD.csv", "w");  
     
	int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        
        grad = gradyanHesaplama(goruntuler, w, grad, vectorSize, trainSize);

        int i;
        for (i = 0; i < vectorSize; i++) {
            w[i] -= eps * grad[i];
        }

        currentLoss = hesaplaKayip(goruntuler, trainSize, w, vectorSize);
        trainLosses[iter] = currentLoss;
        
        saveWeightsToFile("weights_GD.csv", w, vectorSize, iter);

        if (iter > 0 && fabs(currentLoss - previousLoss) < precision) {
            eps *= 0.9; 
        }

        previousLoss = currentLoss;
        basari[iter]=basariOrani(goruntuler,trainSize,w,vectorSize);

        printf("GD Iter %d: Loss = %.6f, Eps = %.6f\n", iter, currentLoss, eps);

        if (eps < 1e-6) {
            printf("Eps çok küçük, eğitim durduruldu.\n");
            break;
        }
    }
    fclose(file);
    return w;
}



float* gradyanHesaplaMiniBatch(goruntu* goruntuler, float* w, float* grad, int vectorsize, int trainsize, int batchsize) {
    int j;
	for (j = 0; j < vectorsize; j++) {
        grad[j] = 0.0;
    }
    int b;
	for (b = 0; b < batchsize; b++) {
        int i = rand() % trainsize; 

        float yTrue = goruntuler[i].sinif;
        float ypred = tanhHesapla(w, goruntuler[i].vektor, vectorsize);

        float sayi = (ypred - yTrue) * (1 - ypred * ypred); 

        int j;
		for (j = 0; j < vectorsize; j++) {
            grad[j] += goruntuler[i].vektor[j] * sayi;
        }
    }
    
    for (j = 0; j < vectorsize; j++) {
        grad[j] = (grad[j]*2)/trainsize;
    }

    return grad;
}

float* stochasticGradientDescentMiniBatch(float* w, float* grad, int vectorSize, int trainSize, goruntu* goruntuler, float* trainLosses,float* basari) {
    float eps = 0.05;           
    float precision = 1e-4;    
    int batchSize = 32;        
    float previousLoss = 1e10; 
    float currentLoss = 0.0;  
    
    FILE* file = fopen("weights_SGD.csv", "w");  
    
    int iter;
    int i;

    for (iter = 0; iter < MAX_ITER; iter++) {
        memset(grad, 0, vectorSize * sizeof(float));

        grad = gradyanHesaplaMiniBatch(goruntuler, w, grad, vectorSize, trainSize, batchSize);

        for (i = 0; i < vectorSize; i++) {
            w[i] -= eps * grad[i]; 
        }

        currentLoss = hesaplaKayip(goruntuler, trainSize, w, vectorSize);
        trainLosses[iter] = currentLoss;
        
        saveWeightsToFile("weights_SGD.csv", w, vectorSize, iter);
        
        basari[iter]=basariOrani(goruntuler,trainSize,w,vectorSize);
        
        printf("SGD Iter %d: Loss = %.6f, PrevLoss = %.6f, Eps = %.6f\n", iter, currentLoss, previousLoss, eps);

        if (fabs(currentLoss - previousLoss) < precision) {
            eps *= 0.9; 
        }

        if (eps < 1e-6) {
            printf("Eps çok küçük. Eğitim durduruldu.\n");
            break;
        }

        previousLoss = currentLoss;
    }
	fclose(file);
    return w;
}

float* adam(float* w, int vectorSize, int trainSize, float* grad, goruntu* goruntuler, float* trainLosses,float* basari) {
    float b1 = 0.9, b2 = 0.999, e = 1e-8; 
    float eps = 0.001;                    
    float precision = 1e-5;              
    float previousLoss = 1e10;           
    float currentLoss = 0.0;          
    int batchSize = 32;                  
    
    
    FILE* file = fopen("weights_Adam.csv", "w");  

    float* m = (float*)calloc(vectorSize, sizeof(float)); 
    float* v = (float*)calloc(vectorSize, sizeof(float)); 

    int t;
	for (t = 1; t <= MAX_ITER; t++) {
       
        grad = gradyanHesaplaMiniBatch(goruntuler, w, grad, vectorSize, trainSize, batchSize);

        int i;
        for (i = 0; i < vectorSize; i++) {
            m[i] = b1 * m[i] + (1 - b1) * grad[i];              
            v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];    
            float mhat = m[i] / (1 - pow(b1, t));              
            float vhat = v[i] / (1 - pow(b2, t));               
            w[i] -= eps * mhat / (sqrt(vhat) + e);            
        }

        
        currentLoss = hesaplaKayip(goruntuler, trainSize, w, vectorSize);
        trainLosses[t - 1] = currentLoss;
        
        saveWeightsToFile("weights_Adam.csv", w, vectorSize, t);
        
        basari[t]=basariOrani(goruntuler,trainSize,w,vectorSize);

        printf("ADAM Iter %d: Loss = %.6f, PrevLoss = %.6f, Eps = %.6f\n", t, currentLoss, previousLoss, eps);

        if (fabs(currentLoss - previousLoss) < precision) {
            printf("Eğitim durdu.\n");
            break;
        }
        previousLoss = currentLoss;
    }
fclose(file);
    free(m);
    free(v);

    return w;
}

void saveWeightsToFile(const char* filename, float* weights, int vectorSize, int iter) {
    FILE* file = fopen(filename, "a");  // "a" ile dosyaya ekleme yapılır
    if (!file) {
        printf("Dosya açılamadı: %s\n", filename);
        return;
    }

    fprintf(file, "%d,", iter);  // İterasyon numarasını yaz
    int i;
    for (i = 0; i < vectorSize; i++) {
        fprintf(file, "%.6f", weights[i]);
        if (i < vectorSize - 1) {
            fprintf(file, ",");  // Son sütunda virgül ekleme
        }
    }
    fprintf(file, "\n");  // Her iterasyon sonrası yeni satır
    fclose(file);
}


void saveLossToFile(const char* filename, int* epochs, float* losses, int size) {
    FILE* file = fopen(filename, "w");
    fprintf(file, "Epoch,Loss\n");
    int i;
    for (i = 0; i < size; i++) {
        fprintf(file, "%d,%.6f\n", epochs[i], losses[i]);
    }
    fclose(file);
}

void saveTestLossToFile(const char* filename, int* epochs,float* basari, int size) {
    FILE* file = fopen(filename, "w");
    fprintf(file, "Epoch,Accuracy\n");
    int i;
	for (i = 0; i < size; i++) {
        fprintf(file, "%d,%.6f\n", epochs[i], basari[i]);
    }
    fclose(file);
}

void saveAccToFile(const char* filename, int* epochs, float* basari, int size) {
    FILE* file = fopen(filename, "w");
    fprintf(file, "Epoch,Accuracy\n");
    int i;
    for (i = 0; i < size; i++) {
        fprintf(file, "%d,%.6f\n", epochs[i], basari[i]);
    }
    fclose(file);
}

void rastgeleAgirlikOlustur(float* w, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        w[i] = ((float)rand() / RAND_MAX) * 0.01; 
    }
}


int main() {
    const char* filenames[] = { 
"circle_1.pgm","triangle_1.pgm",
"circle_2.pgm","triangle_2.pgm",
"circle_3.pgm","triangle_3.pgm",
"circle_4.pgm","triangle_4.pgm",
"circle_5.pgm","triangle_5.pgm",
"circle_6.pgm","triangle_6.pgm",
"circle_7.pgm","triangle_7.pgm",
"circle_8.pgm","triangle_8.pgm",
"circle_9.pgm","triangle_9.pgm",
"circle_10.pgm","triangle_10.pgm",
"circle_11.pgm","triangle_11.pgm",
"circle_12.pgm","triangle_12.pgm",
"circle_13.pgm","triangle_13.pgm",
"circle_14.pgm","triangle_14.pgm",
"circle_15.pgm","triangle_15.pgm",
"circle_16.pgm","triangle_16.pgm",
"circle_17.pgm","triangle_17.pgm",
"circle_18.pgm","triangle_18.pgm",
"circle_19.pgm","triangle_19.pgm",
"circle_20.pgm","triangle_20.pgm",
"circle_21.pgm","triangle_21.pgm",
"circle_22.pgm","triangle_22.pgm",
"circle_23.pgm","triangle_23.pgm",
"circle_24.pgm","triangle_24.pgm",
"circle_25.pgm","triangle_25.pgm",
"circle_26.pgm","triangle_26.pgm",
"circle_27.pgm","triangle_27.pgm",
"circle_28.pgm","triangle_28.pgm",
"circle_29.pgm","triangle_29.pgm",
"circle_30.pgm","triangle_30.pgm",
"circle_31.pgm","triangle_31.pgm",
"circle_32.pgm","triangle_32.pgm",
"circle_33.pgm","triangle_33.pgm",
"circle_34.pgm","triangle_34.pgm",
"circle_35.pgm","triangle_35.pgm",
"circle_36.pgm","triangle_36.pgm",
"circle_37.pgm","triangle_37.pgm",
"circle_38.pgm","triangle_38.pgm",
"circle_39.pgm","triangle_39.pgm",
"circle_40.pgm","triangle_40.pgm",
"circle_41.pgm","triangle_41.pgm",
"circle_42.pgm","triangle_42.pgm",
"circle_43.pgm","triangle_43.pgm",
"circle_44.pgm","triangle_44.pgm",
"circle_45.pgm","triangle_45.pgm",
"circle_46.pgm","triangle_46.pgm",
"circle_47.pgm","triangle_47.pgm",
"circle_48.pgm","triangle_48.pgm",
"circle_49.pgm","triangle_49.pgm",
"circle_50.pgm","triangle_50.pgm",
"circle_51.pgm","triangle_51.pgm",
"circle_52.pgm","triangle_52.pgm",
"circle_53.pgm","triangle_53.pgm",
"circle_54.pgm","triangle_54.pgm",
"circle_55.pgm","triangle_55.pgm",
"circle_56.pgm","triangle_56.pgm",
"circle_57.pgm","triangle_57.pgm",
"circle_58.pgm","triangle_58.pgm",
"circle_59.pgm","triangle_59.pgm",
"circle_60.pgm","triangle_60.pgm",
"circle_61.pgm","triangle_61.pgm",
"circle_62.pgm","triangle_62.pgm",
"circle_63.pgm","triangle_63.pgm",
"circle_64.pgm","triangle_64.pgm",
"circle_65.pgm","triangle_65.pgm",
"circle_66.pgm","triangle_66.pgm",
"circle_67.pgm","triangle_67.pgm",
"circle_68.pgm","triangle_68.pgm",
"circle_69.pgm","triangle_69.pgm",
"circle_70.pgm","triangle_70.pgm",
"circle_71.pgm","triangle_71.pgm",
"circle_72.pgm","triangle_72.pgm",
"circle_73.pgm","triangle_73.pgm",
"circle_74.pgm","triangle_74.pgm",
"circle_75.pgm","triangle_75.pgm",
"circle_76.pgm","triangle_76.pgm",
"circle_77.pgm","triangle_77.pgm",
"circle_78.pgm","triangle_78.pgm",
"circle_79.pgm","triangle_79.pgm",
"circle_80.pgm","triangle_80.pgm",
"circle_81.pgm","triangle_81.pgm",
"circle_82.pgm","triangle_82.pgm",
"circle_83.pgm","triangle_83.pgm",
"circle_84.pgm","triangle_84.pgm",
"circle_85.pgm","triangle_85.pgm",
"circle_86.pgm","triangle_86.pgm",
"circle_87.pgm","triangle_87.pgm",
"circle_88.pgm","triangle_88.pgm",
"circle_89.pgm","triangle_89.pgm",
"circle_90.pgm","triangle_90.pgm",
"circle_91.pgm","triangle_91.pgm",
"circle_92.pgm","triangle_92.pgm",
"circle_93.pgm","triangle_93.pgm",
"circle_94.pgm","triangle_94.pgm",
"circle_95.pgm","triangle_95.pgm",
"circle_96.pgm","triangle_96.pgm",
"circle_97.pgm","triangle_97.pgm",
"circle_98.pgm","triangle_98.pgm",
"circle_99.pgm","triangle_99.pgm",
"circle_100.pgm","triangle_100.pgm",
"circle_101.pgm","triangle_101.pgm",
"circle_102.pgm","triangle_102.pgm",
"circle_103.pgm","triangle_103.pgm",
"circle_104.pgm","triangle_104.pgm",
"circle_105.pgm","triangle_105.pgm",
"circle_106.pgm","triangle_106.pgm",
"circle_107.pgm","triangle_107.pgm",
"circle_108.pgm","triangle_108.pgm",
"circle_109.pgm","triangle_109.pgm",
"circle_110.pgm","triangle_110.pgm",
"circle_111.pgm","triangle_111.pgm",
"circle_112.pgm","triangle_112.pgm",
"circle_113.pgm","triangle_113.pgm",
"circle_114.pgm","triangle_114.pgm",
"circle_115.pgm","triangle_115.pgm",
"circle_116.pgm","triangle_116.pgm",
"circle_117.pgm","triangle_117.pgm",
"circle_118.pgm","triangle_118.pgm",
"circle_119.pgm","triangle_119.pgm",
"circle_120.pgm","triangle_120.pgm",
"circle_121.pgm","triangle_121.pgm",
"circle_122.pgm","triangle_122.pgm",
"circle_123.pgm","triangle_123.pgm",
"circle_124.pgm","triangle_124.pgm",
"circle_125.pgm","triangle_125.pgm",
"circle_126.pgm","triangle_126.pgm",
"circle_127.pgm","triangle_127.pgm",
"circle_128.pgm","triangle_128.pgm",
"circle_129.pgm","triangle_129.pgm",
"circle_130.pgm","triangle_130.pgm",
"circle_131.pgm","triangle_131.pgm",
"circle_132.pgm","triangle_132.pgm",
"circle_133.pgm","triangle_133.pgm",
"circle_134.pgm","triangle_134.pgm",
"circle_135.pgm","triangle_135.pgm",
"circle_136.pgm","triangle_136.pgm",
"circle_137.pgm","triangle_137.pgm",
"circle_138.pgm","triangle_138.pgm",
"circle_139.pgm","triangle_139.pgm",
"circle_140.pgm","triangle_140.pgm",
"circle_141.pgm","triangle_141.pgm",
"circle_142.pgm","triangle_142.pgm",
"circle_143.pgm","triangle_143.pgm",
"circle_144.pgm","triangle_144.pgm",
"circle_145.pgm","triangle_145.pgm",
"circle_146.pgm","triangle_146.pgm",
"circle_147.pgm","triangle_147.pgm",
"circle_148.pgm","triangle_148.pgm",
"circle_149.pgm","triangle_149.pgm",
"circle_150.pgm","triangle_150.pgm",
"circle_151.pgm","triangle_151.pgm",
"circle_152.pgm","triangle_152.pgm",
"circle_153.pgm","triangle_153.pgm",
"circle_154.pgm","triangle_154.pgm",
"circle_155.pgm","triangle_155.pgm",
"circle_156.pgm","triangle_156.pgm",
"circle_157.pgm","triangle_157.pgm",
"circle_158.pgm","triangle_158.pgm",
"circle_159.pgm","triangle_159.pgm",
"circle_160.pgm","triangle_160.pgm",
"circle_161.pgm","triangle_161.pgm",
"circle_162.pgm","triangle_162.pgm",
"circle_163.pgm","triangle_163.pgm",
"circle_164.pgm","triangle_164.pgm",
"circle_165.pgm","triangle_165.pgm",
"circle_166.pgm","triangle_166.pgm",
"circle_167.pgm","triangle_167.pgm",
"circle_168.pgm","triangle_168.pgm",
"circle_169.pgm","triangle_169.pgm",
"circle_170.pgm","triangle_170.pgm",
"circle_171.pgm","triangle_171.pgm",
"circle_172.pgm","triangle_172.pgm",
"circle_173.pgm","triangle_173.pgm",
"circle_174.pgm","triangle_174.pgm",
"circle_175.pgm","triangle_175.pgm",
"circle_176.pgm","triangle_176.pgm",
"circle_177.pgm","triangle_177.pgm",
"circle_178.pgm","triangle_178.pgm",
"circle_179.pgm","triangle_179.pgm",
"circle_180.pgm","triangle_180.pgm",
"circle_181.pgm","triangle_181.pgm",
"circle_182.pgm","triangle_182.pgm",
"circle_183.pgm","triangle_183.pgm",
"circle_184.pgm","triangle_184.pgm",
"circle_185.pgm","triangle_185.pgm",
"circle_186.pgm","triangle_186.pgm",
"circle_187.pgm","triangle_187.pgm",
"circle_188.pgm","triangle_188.pgm",
"circle_189.pgm","triangle_189.pgm",
"circle_190.pgm","triangle_190.pgm",
"circle_191.pgm","triangle_191.pgm",
"circle_192.pgm","triangle_192.pgm",
"circle_193.pgm","triangle_193.pgm",
"circle_194.pgm","triangle_194.pgm",
"circle_195.pgm","triangle_195.pgm",
"circle_196.pgm","triangle_196.pgm",
"circle_197.pgm","triangle_197.pgm",
"circle_198.pgm","triangle_198.pgm",
"circle_199.pgm","triangle_199.pgm",
"circle_200.pgm","triangle_200.pgm",
"circle_201.pgm","triangle_201.pgm",
"circle_202.pgm","triangle_202.pgm",
"circle_203.pgm","triangle_203.pgm",
"circle_204.pgm","triangle_204.pgm",
"circle_205.pgm","triangle_205.pgm",
"circle_206.pgm","triangle_206.pgm",
"circle_207.pgm","triangle_207.pgm",
"circle_208.pgm","triangle_208.pgm",
"circle_209.pgm","triangle_209.pgm",
"circle_210.pgm","triangle_210.pgm",
"circle_211.pgm","triangle_211.pgm",
"circle_212.pgm","triangle_212.pgm",
"circle_213.pgm","triangle_213.pgm",
"circle_214.pgm","triangle_214.pgm",
"circle_215.pgm","triangle_215.pgm",
"circle_216.pgm","triangle_216.pgm",
"circle_217.pgm","triangle_217.pgm",
"circle_218.pgm","triangle_218.pgm",
"circle_219.pgm","triangle_219.pgm",
"circle_220.pgm","triangle_220.pgm",
"circle_221.pgm","triangle_221.pgm",
"circle_222.pgm","triangle_222.pgm",
"circle_223.pgm","triangle_223.pgm",
"circle_224.pgm","triangle_224.pgm",
"circle_225.pgm","triangle_225.pgm",
"circle_226.pgm","triangle_226.pgm",
"circle_227.pgm","triangle_227.pgm",
"circle_228.pgm","triangle_228.pgm",
"circle_229.pgm","triangle_229.pgm",
"circle_230.pgm","triangle_230.pgm",
"circle_231.pgm","triangle_231.pgm",
"circle_232.pgm","triangle_232.pgm",
"circle_233.pgm","triangle_233.pgm",
"circle_234.pgm","triangle_234.pgm",
"circle_235.pgm","triangle_235.pgm",
"circle_236.pgm","triangle_236.pgm",
"circle_237.pgm","triangle_237.pgm",
"circle_238.pgm","triangle_238.pgm",
"circle_239.pgm","triangle_239.pgm",
"circle_240.pgm","triangle_240.pgm",
"circle_241.pgm","triangle_241.pgm",
"circle_242.pgm","triangle_242.pgm",
"circle_243.pgm","triangle_243.pgm",
"circle_244.pgm","triangle_244.pgm",
"circle_245.pgm","triangle_245.pgm",
"circle_246.pgm","triangle_246.pgm",
"circle_247.pgm","triangle_247.pgm",
"circle_248.pgm","triangle_248.pgm",
"circle_249.pgm","triangle_249.pgm",
"circle_250.pgm","triangle_250.pgm",
"circle_251.pgm","triangle_251.pgm",
"circle_252.pgm","triangle_252.pgm",
"circle_253.pgm","triangle_253.pgm",
"circle_254.pgm","triangle_254.pgm",
"circle_255.pgm","triangle_255.pgm",
"circle_256.pgm","triangle_256.pgm",
"circle_257.pgm","triangle_257.pgm",
"circle_258.pgm","triangle_258.pgm",
"circle_259.pgm","triangle_259.pgm",
"circle_260.pgm","triangle_260.pgm",
"circle_261.pgm","triangle_261.pgm",
"circle_262.pgm","triangle_262.pgm",
"circle_263.pgm","triangle_263.pgm",
"circle_264.pgm","triangle_264.pgm",
"circle_265.pgm","triangle_265.pgm",
"circle_266.pgm","triangle_266.pgm",
"circle_267.pgm","triangle_267.pgm",
"circle_268.pgm","triangle_268.pgm",
"circle_269.pgm","triangle_269.pgm",
"circle_270.pgm","triangle_270.pgm",
"circle_271.pgm","triangle_271.pgm",
"circle_272.pgm","triangle_272.pgm",
"circle_273.pgm","triangle_273.pgm",
"circle_274.pgm","triangle_274.pgm",
"circle_275.pgm","triangle_275.pgm",
"circle_276.pgm","triangle_276.pgm",
"circle_277.pgm","triangle_277.pgm",
"circle_278.pgm","triangle_278.pgm",
"circle_279.pgm","triangle_279.pgm",
"circle_280.pgm","triangle_280.pgm",
"circle_281.pgm","triangle_281.pgm",
"circle_282.pgm","triangle_282.pgm",
"circle_283.pgm","triangle_283.pgm",
"circle_284.pgm","triangle_284.pgm",
"circle_285.pgm","triangle_285.pgm",
"circle_286.pgm","triangle_286.pgm",
"circle_287.pgm","triangle_287.pgm",
"circle_288.pgm","triangle_288.pgm",
"circle_289.pgm","triangle_289.pgm",
"circle_290.pgm","triangle_290.pgm",
"circle_291.pgm","triangle_291.pgm",
"circle_292.pgm","triangle_292.pgm",
"circle_293.pgm","triangle_293.pgm",
"circle_294.pgm","triangle_294.pgm",
"circle_295.pgm","triangle_295.pgm",
"circle_296.pgm","triangle_296.pgm",
"circle_297.pgm","triangle_297.pgm",
"circle_298.pgm","triangle_298.pgm",
"circle_299.pgm","triangle_299.pgm",
"circle_300.pgm","triangle_300.pgm",
"circle_301.pgm","triangle_301.pgm",
"circle_302.pgm","triangle_302.pgm",
"circle_303.pgm","triangle_303.pgm",
"circle_304.pgm","triangle_304.pgm",
"circle_305.pgm","triangle_305.pgm",
"circle_306.pgm","triangle_306.pgm",
"circle_307.pgm","triangle_307.pgm",
"circle_308.pgm","triangle_308.pgm",
"circle_309.pgm","triangle_309.pgm",
"circle_310.pgm","triangle_310.pgm",
"circle_311.pgm","triangle_311.pgm",
"circle_312.pgm","triangle_312.pgm",
"circle_313.pgm","triangle_313.pgm",
"circle_314.pgm","triangle_314.pgm",
"circle_315.pgm","triangle_315.pgm",
"circle_316.pgm","triangle_316.pgm",
"circle_317.pgm","triangle_317.pgm",
"circle_318.pgm","triangle_318.pgm",
"circle_319.pgm","triangle_319.pgm",
"circle_320.pgm","triangle_320.pgm",
"circle_321.pgm","triangle_321.pgm",
"circle_322.pgm","triangle_322.pgm",
"circle_323.pgm","triangle_323.pgm",
"circle_324.pgm","triangle_324.pgm",
"circle_325.pgm","triangle_325.pgm",
"circle_326.pgm","triangle_326.pgm",
"circle_327.pgm","triangle_327.pgm",
"circle_328.pgm","triangle_328.pgm",
"circle_329.pgm","triangle_329.pgm",
"circle_330.pgm","triangle_330.pgm",
"circle_331.pgm","triangle_331.pgm",
"circle_332.pgm","triangle_332.pgm",
"circle_333.pgm","triangle_333.pgm",
"circle_334.pgm","triangle_334.pgm",
"circle_335.pgm","triangle_335.pgm",
"circle_336.pgm","triangle_336.pgm",
"circle_337.pgm","triangle_337.pgm",
"circle_338.pgm","triangle_338.pgm",
"circle_339.pgm","triangle_339.pgm",
"circle_340.pgm","triangle_340.pgm",
"circle_341.pgm","triangle_341.pgm",
"circle_342.pgm","triangle_342.pgm",
"circle_343.pgm","triangle_343.pgm",
"circle_344.pgm","triangle_344.pgm",
"circle_345.pgm","triangle_345.pgm",
"circle_346.pgm","triangle_346.pgm",
"circle_347.pgm","triangle_347.pgm",
"circle_348.pgm","triangle_348.pgm",
"circle_349.pgm","triangle_349.pgm",
"circle_350.pgm","triangle_350.pgm",
"circle_351.pgm","triangle_351.pgm",
"circle_352.pgm","triangle_352.pgm",
"circle_353.pgm","triangle_353.pgm",
"circle_354.pgm","triangle_354.pgm",
"circle_355.pgm","triangle_355.pgm",
"circle_356.pgm","triangle_356.pgm",
"circle_357.pgm","triangle_357.pgm",
"circle_358.pgm","triangle_358.pgm",
"circle_359.pgm","triangle_359.pgm",
"circle_360.pgm","triangle_360.pgm",
"circle_361.pgm","triangle_361.pgm",
"circle_362.pgm","triangle_362.pgm",
"circle_363.pgm","triangle_363.pgm",
"circle_364.pgm","triangle_364.pgm",
"circle_365.pgm","triangle_365.pgm",
"circle_366.pgm","triangle_366.pgm",
"circle_367.pgm","triangle_367.pgm",
"circle_368.pgm","triangle_368.pgm",
"circle_369.pgm","triangle_369.pgm",
"circle_370.pgm","triangle_370.pgm",
"circle_371.pgm","triangle_371.pgm",
"circle_372.pgm","triangle_372.pgm",
"circle_373.pgm","triangle_373.pgm",
"circle_374.pgm","triangle_374.pgm",
"circle_375.pgm","triangle_375.pgm",
"circle_376.pgm","triangle_376.pgm",
"circle_377.pgm","triangle_377.pgm",
"circle_378.pgm","triangle_378.pgm",
"circle_379.pgm","triangle_379.pgm",
"circle_380.pgm","triangle_380.pgm",
"circle_381.pgm","triangle_381.pgm",
"circle_382.pgm","triangle_382.pgm",
"circle_383.pgm","triangle_383.pgm",
"circle_384.pgm","triangle_384.pgm",
"circle_385.pgm","triangle_385.pgm",
"circle_386.pgm","triangle_386.pgm",
"circle_387.pgm","triangle_387.pgm",
"circle_388.pgm","triangle_388.pgm",
"circle_389.pgm","triangle_389.pgm",
"circle_390.pgm","triangle_390.pgm",
"circle_391.pgm","triangle_391.pgm",
"circle_392.pgm","triangle_392.pgm",
"circle_393.pgm","triangle_393.pgm",
"circle_394.pgm","triangle_394.pgm",
"circle_395.pgm","triangle_395.pgm",
"circle_396.pgm","triangle_396.pgm",
"circle_397.pgm","triangle_397.pgm",
"circle_398.pgm","triangle_398.pgm",
"circle_399.pgm","triangle_399.pgm",
"circle_400.pgm","triangle_400.pgm",
"circle_401.pgm","triangle_401.pgm",
"circle_402.pgm","triangle_402.pgm",
"circle_403.pgm","triangle_403.pgm",
"circle_404.pgm","triangle_404.pgm",
"circle_405.pgm","triangle_405.pgm",
"circle_406.pgm","triangle_406.pgm",
"circle_407.pgm","triangle_407.pgm",
"circle_408.pgm","triangle_408.pgm",
"circle_409.pgm","triangle_409.pgm",
"circle_410.pgm","triangle_410.pgm",
"circle_411.pgm","triangle_411.pgm",
"circle_412.pgm","triangle_412.pgm",
"circle_413.pgm","triangle_413.pgm",
"circle_414.pgm","triangle_414.pgm",
"circle_415.pgm","triangle_415.pgm",
"circle_416.pgm","triangle_416.pgm",
"circle_417.pgm","triangle_417.pgm",
"circle_418.pgm","triangle_418.pgm",
"circle_419.pgm","triangle_419.pgm",
"circle_420.pgm","triangle_420.pgm",
"circle_421.pgm","triangle_421.pgm",
"circle_422.pgm","triangle_422.pgm",
"circle_423.pgm","triangle_423.pgm",
"circle_424.pgm","triangle_424.pgm",
"circle_425.pgm","triangle_425.pgm",
"circle_426.pgm","triangle_426.pgm",
"circle_427.pgm","triangle_427.pgm",
"circle_428.pgm","triangle_428.pgm",
"circle_429.pgm","triangle_429.pgm",
"circle_430.pgm","triangle_430.pgm",
"circle_431.pgm","triangle_431.pgm",
"circle_432.pgm","triangle_432.pgm",
"circle_433.pgm","triangle_433.pgm",
"circle_434.pgm","triangle_434.pgm",
"circle_435.pgm","triangle_435.pgm",
"circle_436.pgm","triangle_436.pgm",
"circle_437.pgm","triangle_437.pgm",
"circle_438.pgm","triangle_438.pgm",
"circle_439.pgm","triangle_439.pgm",
"circle_440.pgm","triangle_440.pgm",
"circle_441.pgm","triangle_441.pgm",
"circle_442.pgm","triangle_442.pgm",
"circle_443.pgm","triangle_443.pgm",
"circle_444.pgm","triangle_444.pgm",
"circle_445.pgm","triangle_445.pgm",
"circle_446.pgm","triangle_446.pgm",
"circle_447.pgm","triangle_447.pgm",
"circle_448.pgm","triangle_448.pgm",
"circle_449.pgm","triangle_449.pgm",
"circle_450.pgm","triangle_450.pgm",
"circle_451.pgm","triangle_451.pgm",
"circle_452.pgm","triangle_452.pgm",
"circle_453.pgm","triangle_453.pgm",
"circle_454.pgm","triangle_454.pgm",
"circle_455.pgm","triangle_455.pgm",
"circle_456.pgm","triangle_456.pgm",
"circle_457.pgm","triangle_457.pgm",
"circle_458.pgm","triangle_458.pgm",
"circle_459.pgm","triangle_459.pgm",
"circle_460.pgm","triangle_460.pgm",
"circle_461.pgm","triangle_461.pgm",
"circle_462.pgm","triangle_462.pgm",
"circle_463.pgm","triangle_463.pgm",
"circle_464.pgm","triangle_464.pgm",
"circle_465.pgm","triangle_465.pgm",
"circle_466.pgm","triangle_466.pgm",
"circle_467.pgm","triangle_467.pgm",
"circle_468.pgm","triangle_468.pgm",
"circle_469.pgm","triangle_469.pgm",
"circle_470.pgm","triangle_470.pgm",
"circle_471.pgm","triangle_471.pgm",
"circle_472.pgm","triangle_472.pgm",
"circle_473.pgm","triangle_473.pgm",
"circle_474.pgm","triangle_474.pgm",
"circle_475.pgm","triangle_475.pgm",
"circle_476.pgm","triangle_476.pgm",
"circle_477.pgm","triangle_477.pgm",
"circle_478.pgm","triangle_478.pgm",
"circle_479.pgm","triangle_479.pgm",
"circle_480.pgm","triangle_480.pgm",
"circle_481.pgm","triangle_481.pgm",
"circle_482.pgm","triangle_482.pgm",
"circle_483.pgm","triangle_483.pgm",
"circle_484.pgm","triangle_484.pgm",
"circle_485.pgm","triangle_485.pgm",
"circle_486.pgm","triangle_486.pgm",
"circle_487.pgm","triangle_487.pgm",
"circle_488.pgm","triangle_488.pgm",
"circle_489.pgm","triangle_489.pgm",
"circle_490.pgm","triangle_490.pgm",
"circle_491.pgm","triangle_491.pgm",
"circle_492.pgm","triangle_492.pgm",
"circle_493.pgm","triangle_493.pgm",
"circle_494.pgm","triangle_494.pgm",
"circle_495.pgm","triangle_495.pgm",
"circle_496.pgm","triangle_496.pgm",
"circle_497.pgm","triangle_497.pgm",
"circle_498.pgm","triangle_498.pgm",
"circle_499.pgm","triangle_499.pgm",
"circle_500.pgm","triangle_500.pgm"
     };
    srand(time(NULL));

    char gdAccuracyFile[50];
    char gdLossFile[50];
    char sgdAccuracyFile[50];
    char sgdLossFile[50];
    char adamAccuracyFile[50];
    char adamLossFile[50];


    int numFiles = sizeof(filenames) / sizeof(filenames[0]);
    int width, height;
    goruntu goruntuler[numFiles];

    int vectorSize = 0;
	int i,j;
	int iter;


    for (i = 0; i < numFiles; i++) {
        float* vector = readPGMImage(filenames[i], &width, &height);
        if (!vector) continue;
        vectorSize = width * height + 1;
        goruntuler[i].vektor = vector;
        if(filenames[i][0] == 'c'){
            goruntuler[i].sinif =1;
        }
        else{
            goruntuler[i].sinif = -1;
        }
    }

    int testSize = numFiles / 5;
    int trainSize = numFiles - testSize;
    goruntu egitimKumesi[trainSize];
    goruntu testKumesi[testSize];

    bolEgitimTestKumesi(goruntuler,numFiles,egitimKumesi,testKumesi);

    float* w = (float*)malloc(vectorSize * sizeof(float));
    float* grad = (float*)calloc(vectorSize, sizeof(float));
    int epochs[MAX_ITER];
    float trainLosses[MAX_ITER] = {0};
    float testLosses[MAX_ITER] = {0};
    float basari[MAX_ITER]={0};

    for (i = 0; i < MAX_ITER; i++) {
        epochs[i] = i + 1;
    }


    for ( j = 0; j < 5; j++) {

        // GD
        sprintf(gdAccuracyFile, "accuracy_GD_%d.csv", j + 1);
        sprintf(gdLossFile, "train_loss_GD_%d.csv", j + 1);
        rastgeleAgirlikOlustur(w, vectorSize);
        memset(grad, 0, vectorSize * sizeof(float));
        w = gradientDescent(w, grad, vectorSize, trainSize, egitimKumesi, trainLosses, basari);
        saveLossToFile(gdLossFile, epochs, trainLosses, MAX_ITER);
        saveAccToFile(gdAccuracyFile, epochs, basari, MAX_ITER);

        // SGD
        sprintf(sgdAccuracyFile, "accuracy_SGD_%d.csv", j + 1);
        sprintf(sgdLossFile, "train_loss_SGD_%d.csv", j + 1);
        rastgeleAgirlikOlustur(w, vectorSize);
        memset(grad, 0, vectorSize * sizeof(float));
        w = stochasticGradientDescentMiniBatch(w, grad, vectorSize, trainSize, egitimKumesi, trainLosses,basari);
        saveLossToFile(sgdLossFile, epochs, trainLosses, MAX_ITER);
        saveAccToFile(sgdAccuracyFile, epochs, basari, MAX_ITER);

        // Adam
        sprintf(adamAccuracyFile, "accuracy_Adam_%d.csv", j + 1);
        sprintf(adamLossFile, "train_loss_Adam_%d.csv", j + 1);
        rastgeleAgirlikOlustur(w, vectorSize);
        memset(grad, 0, vectorSize * sizeof(float));
        w = adam(w, vectorSize, trainSize, grad, egitimKumesi, trainLosses,basari);
        saveLossToFile(adamLossFile, epochs, trainLosses, MAX_ITER);
        saveAccToFile(adamAccuracyFile, epochs, basari, MAX_ITER);
    }
    
	


    free(w);
    free(grad);
    for (i = 0; i < numFiles; i++) {
        free(goruntuler[i].vektor);
    }

    return 0;
}