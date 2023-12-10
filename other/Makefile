all: rk4.o functions.o chebyshev.o
integration: rk4.o functions.o
cheb: chebyshev.o

rk4.o: rk4.c
	gcc -c -fPIC rk4.c -o rk4.o
	gcc rk4.o -shared -o lib_rk4.so

functions.o: functions.c
	gcc -c -fPIC functions.c -o functions.o
	gcc functions.o -shared -o lib_functions.so

chebyshev.o: chebyshev.c
	gcc -c -fPIC chebyshev.c -o chebyshev.o
	gcc chebyshev.o -shared -o lib_chebyshev.so