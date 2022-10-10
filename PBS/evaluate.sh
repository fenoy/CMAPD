count=50

for i in $(seq 0 $count-1); do
   ./MAPD -m ../instances/maps/$i.map -a ../instances/maps/$i.map -t ../instances/tasks/$i.task -s PP --capacity 2 --objective makespan --only-update-top --kiva -o ./output.txt

    ./pbs -m instances/$i.map -a instances/$0.task -o out.txt --outputPaths=paths.txt -k 20 -t 60
done
