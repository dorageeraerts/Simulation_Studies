:construction: <span style="color:yellow">**Documentation under construction** </span>

MURAVES simulation
========================

# Muraves environment - Docker

The *muraves-sim* container is used to run the MuRaVes simulation. A container can be thought of as a snapshot of a virtual machine: no matter what changes are made while using it, it always starts from the same initial state captured in the snapshot. However, it is possible to connect local folders to the container. These connected folders, called "mounted" folders, are not temporary: any changes made inside these folders from within the container are persistent and visible outside the container as well. This allows you to develop scripts or modify files without losing your work, while still benefiting from a standardized, consistent set of software and packages inside the container.

### Step-by-step: Use the container on T2B

1. Connect to T2B
2. Download last version of setup_container.sh by doing *add command and path*
3. Make it executable `chmod +x setup_container.sh`
4. Execute: `./setup_container.sh`. It will ask for the folder (input directory) where you want/have your Simulation_Studies git repository. By default this is in a directory MURAVES inside your home directory. (If you would like to automatically set this folder, then change the DEFAULT_WORKSPACE path in setup_container.sh to the path where you want/have your git repository.)
5. If succeeded, congratulations! The container is installed and running. The container is up and running is you see this output:

![image](../documentation/WelcomeContainer.png)
  
Software available:
- Geant4: version 11.4
- CRY: version 1.7
- EcoMug

The workspace directory in the container is mounted to the input directory where you want/have git repository on T2B.
This means that, in this mounted directory, changes in scripts will persist, even when leaving the container.

The file setup_container.sh will automatically:
- Run the image `/group/Muography/container/simulation_container/muraves-sim-latest.sif`
- Install Geant4 and other dependencies (CRY, EcoMug).
- Return a welcome message
- Check that in the workspace there is a git reposotory. 
- **important**: The workspace is set to be the MuRaVes Git repository on T2B located in `/group/Muography/Software/`. This is a read-only folder with the only objective to run stable release of the MuRaVes software. This means you cannot modify or create new script here. If the use wish to use the container for developing purposes it is of course possible [Check this instructions](#container-in-developing-mode)!

### Editor: Visual Studio Code
The choice of the editor is personal, read scripts with your favourite editor. However, it can be handy to ssh to T2B directly from your editor and open the scripts as if you were working on your local laptop. For this reason VSCode comes in handy. On VSCode an extension is available called Remote-SSH, provided by Microsoft, that allows you to tunnel directly to T2B.

**NB: Using a container is exactly like using another laptop which has only a few folder in common with T2B:**
  - You can pull and push from Git
  - Changes to scripts are persistent (meaning outside the container)
  BUT:
  - Changes to the software installed in the container (new conda envs, new python packages, new libriaries, etc.) are temporary and exist only during your current session.

## Local installation
*For the local execution of MuRaVes script, the user is free to set up the preferred environment. One possibility is to use the container available in this Git repository: in the `/environment/docker/` folder are available the Dockerfile and a bash script `entrypoint.sh` that are needed to build the image of the container.*

The following steps leads to the installation of the container.
- I don't have a MuRaVes Git repository -> [Steps](#i-do-not-have-a-muraves-repo-yet)

- I have already a the MuRaVes Git repository cloned on my laptop -> [Steps](#i-have-a-muraves-repo).

### I do not have a MuRaVes repo yet
1. Clone the entire repository locally using `git clone`:
   ```bash
   git clone https://github.com/muraves/Simulation_Studies.git 
   ```
2. Enter the folder where the docker file is located `~$ cd Software/environment/docker/`, and run:
   ```bash
   docker build -t muraves-sim .
   ```
3. Now you can run the image and choose the working directory, for instance:
   ```bash
   docker run -it -v <~/my/local/folder>/Simulation_Studies:/workspace muraves-sim:latest
   ```
   By giving the path `<~/my/local/folder>/Simulation_Studies` before `:/workspace`, it means that all the script locally available inside this folder will be available also in the container. **The changes that the user applies while working in the container will persists once the container il closed.**
4. The user is now ready to go: develop or run scripts.

### I have a MuRaVes repo
1. Pull last changes from Git by running `git pull` from the `Simulation_Studies` directory.
2. Enter the folder where the docker file is located `~$ cd Software/environment/docker/`, and run:
   ```bash
   docker build -t muraves-sim .
   ```
2. Now you can run the image and choose the working directory, for instance:
   ```bash
   docker run -it -v <~/my/local/folder>/Simulation_Studies:/workspace muraves-sim:latest
   ```
   By giving the path `<~/my/local/folder>/Simulation_Studies` before `:/workspace`, it means that all the script locally available inside this folder will be available also in the container. **The changes that the user applies while working in the container will persists once the container il closed.**
3. The user is now ready to go: develop or run scripts.

# Container installation - for maintainers
T2B does not support docker. Therefore a few steps are necessary in order to bring the container there.
- Apply the changes to the Dockerfile locally.
- Build the image locally as explained [here](#i-have-a-muraves-repo)
- 
- Tag the image on Dockerhub:
  ```bash
  docker tag muraves-sim:latest <Docker_username>/muraves-sim:latest
  ```
- Push the image on Dockerhub:
  ```bash
  docker push <Docker_username>/muraves-sim:latest
  ```
- Login to T2B and build the image:
  ```bash
  apptainer build muraves-sim.sif docker://dgeeraer/muraves-sim:latest
  ```
- Given that the paths in setup_container.sh are correct (pointing tot the .sif image), the container can be ran with
  ```bash
  ./setup_container.sh
  ```

Compatibility
-------------

Licence
-------

The image is based on an exisiting Geant4 Docker image, created by physino and available on [DockerHub](https://hub.docker.com/r/physino/geant4).

Authors
-------

The file was written by [Dora Geeraerts](mailto:dora.geeraerts@vub.be).
