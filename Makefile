CC=mpicc #compiler
CFLAGS=-O2 -Wall -std=c99 #options

SRC = src
BIN = bin
RESULTS = results

#targets
all: test

test: $(SRC)/test.c $(SRC)/my_mpi.c #target depends on prerequisite
	mkdir -p $(BIN)
	$(CC) $(CFLAGS) $^ -o $(BIN)/test
#					 pre

clean:
	rm -rf $(BIN) $(RESULTS)