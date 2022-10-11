for i in $(seq 0 $(($4-1))); do
    ./pbs -m env/grid.map -a instances/a$2-t$3/ta-$1/$i.assignment -o ../pbs_$1_a$2_t$3.csv -k $2 -t 60
done
