This is a photometric stereo. Using images taken with different 
sources of light, we can calculate the shape of a 3D object.
This code is written to work with the CAT object but can be 
easily adapted to other objects from the PSData dataset, you 
only need to change the paths to the image directories and to
select a new mask.

The program will output selected mask, the gradient space 
coordinates of the normal, the 3D representation of the object
that you can rotate in 3D space and the 2D depth map of the 
object.

The required dependencies are listed in the requirements.txt 
file.

To execute the program, install the dependencies, listed in the
requirements.txt file. Then just run the script after declaring
correct paths.

Alternatively, run the command to make the shell script
executable:

    chmod +x run_project.sh

Run the script using:

    ./run_project.sh


(Didn't test the .sh, might not work, try doing it manually)