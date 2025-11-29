pdpcontrol stop
sleep 10
pdpcontrol start 1
sleep 20
echo "its" | ncat localhost 1025
echo -e "\eG" | ncat localhost 1025
sleep 20
pdp imlac
sleep 10
imlac=$(xdotool search --name 'Imlac - CRT - Imlac Display')
xdotool windowactivate $imlac
#xdotool key 0xff1a
xdotool key ctrl+z
sleep 5
d=200 ###delay between keystrokes
xdotool windowactivate $imlac #redo this each command in case a user clicks away
xdotool type --window $imlac --delay $d ':LOGIN LARS'
xdotool key 0xff0d ###carriage return
sleep 1
xdotool windowactivate $imlac
xdotool type --window $imlac --delay $d ':TCTYPE OIMLAC'
xdotool key 0xff0d
sleep 1
xdotool windowactivate $imlac
xdotool type --window $imlac --delay $d ':CWD GAMES'
xdotool key 0xff0d
sleep 1
xdotool windowactivate $imlac
xdotool type --window $imlac --delay $d ':RUN MAZE C'
xdotool key 0xff0d
xdotool windowactivate $imlac
sleep 2
xdotool windowactivate $imlac
xdotool key 0xff0d #### return to accept default maze
sleep 10
#should now be looking at the map of the maze
xdotool windowactivate $imlac
xdotool key Right
#done. Now in the maze
