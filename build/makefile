all: AddTwoArray maldelbrot test

AddTwoArray: 
	nvc++ ../src/AddTwoArray.cu -o AddTwoArray

maldelbrot:
	nvc++ ../src/maldelbrot.cu -o maldelbrot

test:
	nvc++ ../src/test.cu -o test

clean:
	rm -rf *.o AddTwoArray maldelbrot test