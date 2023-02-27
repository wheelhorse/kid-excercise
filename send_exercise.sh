python3 /auto/exercise_kid/genMath.py 100
tar -czf /tmp/exercise_kid.tar.gz /auto/exercise_kid
echo "Kid Exercise Math" |mail -s "10 Days Kid Math Exercise" -a "From:kidMath@cisco.com" -A "/tmp/exercise_kid.tar.gz" fengbwu@cisco.com ziyma@cisco.com
