# ABB 6640 Setup Instruction
## Safety Locks
* Align magnetic sensors
* Verify all signals green
* Press YELLOW button 

## Ethernet Ports
* **Service Port**: Used when using RobotStudio
* **WAN(X6)**: Used when using python module

## Repo
* [abb_motion_program_exec](https://github.com/johnwason/abb_motion_program_exec):
  ```
  git clone https://github.com/johnwason/abb_motion_program_exec.git
  python setup.py install
  ```
* [Motion-Primitive-Planning](https://github.com/hehonglu123/Motion-Primitive-Planning):
  ```
  git clone https://github.com/hehonglu123/Motion-Primitive-Planning.git
  ```
## RobotStudio

### Activation
*Options* -> *Activation Wizard* ->*Network License* -> *License Server*:  `licsrvr33.win.rpi.edu`

### Controller Setup

Follow readme at [robot_setup_manual](https://github.com/johnwason/abb_motion_program_exec/blob/master/doc/robot_setup_manual.md)

#### Connect to Controller
*Controller* -> *Add Controller* -> *One Click Connect* -> *Log in as Default User*
#### File Transfer
*Controller* -> *File Transfer*
#### Write Access
*Controller* -> *Request Write Access*
#### P-Start (RAPID RESET)
*Controller* -> *Restart* -> Reset RAPID (P-start)

#### IP Settings
* PC IP: *Controller*->*Configuration*->*Communication*->*Transmission Protocol* -> *UCDevice*-> *Remote Device*
* IRC5 IP: *Controller*->*Configuration*->*IP Setting*->*Add (Interface WAN)*

## PC Setting
Switch to Manual IP when using python modules (what specified in *PC IP* above)
