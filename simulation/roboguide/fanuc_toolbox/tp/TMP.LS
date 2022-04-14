/PROG  TMP
/ATTR
/MN
   1:  UFRAME_NUM=0 ;
   2:  UTOOL_NUM=1 ;
   3:  DO[101]=ON ;
   4:  RUN DATARECORDER ;
   5:L P[1] 50mm/sec CNT100 ;
   6:L P[2] 50mm/sec FINE ;
   7:J P[3] 100% FINE ;
   8:L P[4] 50mm/sec CNT100 ;
   9:C P[5] 
    :  P[6] 50mm/sec FINE ;
   10:  DO[101]=OFF ;
/POS
P[1]{
   GP1:
   UF : 0, UT : 1,     CONFIG : 'N U T, 0, 0, 0',
   X = 1850.000 mm,  Y = 200.000 mm,  Z = 290.000 mm,
   W = -180.000 deg,  P = 0.000 deg,  R = 0.000 deg
};
P[2]{
   GP1:
   UF : 0, UT : 1,     CONFIG : 'N U T, 0, 0, 0',
   X = 1850.000 mm,  Y = 200.000 mm,  Z = 589.000 mm,
   W = -180.000 deg,  P = 0.000 deg,  R = 0.000 deg
};
P[3]{
   GP1:
   UF : 0, UT : 1,
   J1 = 0.000 deg,  J2 = 20.000 deg,  J3 = -10.000 deg,
   J4 = 0.000 deg,  J5 = -20.000 deg,  J6 = 10.000 deg
};
P[4]{
   GP1:
   UF : 0, UT : 1,     CONFIG : 'N U T, 0, 0, 0',
   X = 1850.000 mm,  Y = 200.000 mm,  Z = 589.000 mm,
   W = -180.000 deg,  P = 0.000 deg,  R = 0.000 deg
};
P[5]{
   GP1:
   UF : 0, UT : 1,     CONFIG : 'N U T, 0, 0, 0',
   X = 1850.000 mm,  Y = 250.000 mm,  Z = 400.000 mm,
   W = -180.000 deg,  P = 0.000 deg,  R = 0.000 deg
};
P[6]{
   GP1:
   UF : 0, UT : 1,     CONFIG : 'N U T, 0, 0, 0',
   X = 1850.000 mm,  Y = 200.000 mm,  Z = 290.000 mm,
   W = -180.000 deg,  P = 0.000 deg,  R = 0.000 deg
};
/END
