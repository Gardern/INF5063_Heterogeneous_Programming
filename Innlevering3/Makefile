CC = nvcc
CFLAGS = -O3 -g -Xcompiler -fopenmp -arch sm_30
LDFLAGS = -lm

all: c63enc c63dec c63pred

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

c63enc: c63enc.cu dsp.cu tables.cu io.cu c63_write.cu common.cu me.cu
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.cu dsp.cu tables.cu io.cu common.cu me.cu
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.cu dsp.cu tables.cu io.cu common.cu me.cu
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@
	
clean:
	rm -f *.o c63enc c63dec c63pred
