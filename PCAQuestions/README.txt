Project Name:CSCI 57300 Data Mining
Author: Kaiming Cui



Environment
This project is developed in Python 3.6.1 with 1 library in my mac book
My conda enviroment uses Python 3.7.4 to run my project.
Both version of python work.

Dependency:
1. numpy
2. sys
2. numpy.linalg
3. numpy.matlib

Building and Testing:
1. Upload the zip file to tesla server on local (before login).
    scp assign2_iuname.zip [user_name]@tesla.cs.iupui.edu:/home/[user_name]
    * [user_name] is your username in IUPUI.
2. Login to tesla server with your private password.
    ssh [user_name]@tesla.cs.iupui.edu
3. Archive decompression.
    unzip assign2_iuname.zip
4. Go into the file root “assign2_name”.
    cd assign2_iuname
5. Run the script.
    python assign2_name.py argv[1]
    argv[1] means the parameter you input, it represent the file name.

Note:
1. The data file is magic04.data, You send the file name as parameter into my project.
2. After you run my project, you will see the ouput on your screen. In addition, it will also produce a txt file called 'assign2_iuKaimingCui.txt' that will contain all the ouput.
