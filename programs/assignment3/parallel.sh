for i in {1..20}
do
	mpirun -np 2 ./jacobi_parallel
done

for i in {1..20}
do
	mpirun -np 4 ./jacobi_parallel
done

for i in {1..20}
do
	mpirun -np 8 ./jacobi_parallel
done

for i in {1..20}
do
	mpirun -np 16 ./jacobi_parallel
done
