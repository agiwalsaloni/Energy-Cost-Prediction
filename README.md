# Energy-Cost-Prediction
Aim of the project is to provide the user with energy statistics so that he can better plan energy consumption which will help in better cost planning for the user and will pave a road towards sustainable development of the entire planet

User Manual :

The application is a Flask application, coded fully in python. To run the following application, the system needs to have all the required python libraries installed on the local machine.
 
It is advisable to work in the virtual environment, to avoid the version issues later.
Steps to create a virtual environment :
1. First, install pip for python3 :
		python3 -m pip install --user --upgrade pip
2. Now installing the virtual environment :
On macOS and Linux:
python3 -m pip install --user virtualenv
On Windows:
py -m pip install --user virtualenv
3. Creating the virtual environment :
	On macOS and Linux:
python3 -m venv env
On Windows:
py -m venv env
4. Activating the virtual environment :
	On macOS and Linux:
source env/bin/activate
On Windows:
.\env\Scripts\activate
 
Now for installing the libraries, just run the command :
pip install -r requirements.txt
4.	 Now after installing all the required libraries, the application is ready to run. The application is in development state as for now, so it has to be run on the local host i.e. the local system on which the application is being hosted. 
To run the application, run the command:
python  api1.py 
5.	After running the above command, open your web browser and enter the URL which appears on your console. 
6.	Your application is ready to be explored.
