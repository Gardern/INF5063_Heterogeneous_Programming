#PPUCC = ppu-gcc
#SPUCC = spu-gcc
#CFLAGS_PPU_ENC = -O3 -Wall -g -lspe2
#CFLAGS_PPU_DEC = -m32 -O3 -Wall -g
#CFLAGS_SPU = -O3 -Wall -g
#LDFLAGS = -lm
#
#all: spetest c63enc c63dec c63pred
#
#%.o: %.c
#	$(CC) $< $(CFLAGS) -c -o $@
#
#spetest: spetest.c
#	$(SPUCC) $^ $(CFLAGS_SPU) $(LDFLAGS) -o $@
#c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
#	$(PPUCC) $^ $(CFLAGS_PPU_ENC) $(LDFLAGS) -o $@
#c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(PPUCC) $^ $(CFLAGS_PPU_DEC) $(LDFLAGS) -o $@
#c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(PPUCC) $^ -DC63_PRED $(CFLAGS_PPU_DEC) $(LDFLAGS) -o $@
#
#clean:
#	rm -f *.o c63enc c63dec c63pred

#CC = ppu-gcc
#SPUCC = spu-gcc
#CFLAGS = -O3 -Wall -g -lspe2
#DEPS = c63.h tables.h

all: c63enc meblockspe dctquantizespe dequantizeidctspe c63dec
#%.o: %.c $(DEPS)
#	$(CC) -c -o $@ $< $(CFLAGS)

c63enc: c63enc.c dsp.c tables.c io.c c63_write.c c63.h common.c me.c
	ppu-gcc -o c63enc c63enc.c dsp.c tables.c io.c c63_write.c common.c me.c -O3 -Wall -O -g -lsimdmath -maltivec -lspe2 -lm
meblockspe: me_block_spe.c
	spu-gcc -o meblockspe me_block_spe.c -O3 -Wall -O -g -lsimdmath -lm
dctquantizespe: dct_quantize_spe.c
	spu-gcc -o dctquantizespe dct_quantize_spe.c -O3 -Wall -g -lsimdmath -lm
dequantizeidctspe: dequantize_idct_spe.c
	spu-gcc -o dequantizeidctspe dequantize_idct_spe.c -O3 -Wall -g -lsimdmath -lm
c63dec: c63dec.c dsp.c tables.c io.c c63.h common.c me.c
	ppu-gcc -o c63dec c63dec.c dsp.c tables.c io.c c63_write.c common.c me.c -m32 -O3 -Wall -O -g -lsimdmath -maltivec -lspe2 -lm

clean:
	rm -f *.o c63enc meblockspe dctquantizespe dequantizeidctspe c63dec
