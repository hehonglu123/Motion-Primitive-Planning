/PROG  TMP
/ATTR
/MN
   1:  UFRAME_NUM=1 ;
   2:  UTOOL_NUM=2 ;
   3:  DO[101]=ON ;
   4:  RUN DATARECORDER ;
   5:J P[1] 50% FINE ;
   6:L P[2] 5mm/sec FINE ;
   7:  DO[101]=OFF ;
/POS
P[1]{
   GP1:
   UF : 1, UT : 2,
   J1 = 18.642 deg,  J2 = 8.336 deg,  J3 = -43.694 deg,
   J4 = -34.747 deg,  J5 = 32.802 deg,  J6 = 25.254 deg
};
P[2]{
   GP1:
   UF : 1, UT : 2,     CONFIG : 'F U T, 0, 0, 0',
   X = 2199.200 mm,  Y = 500.000 mm,  Z = 1000.000 mm,
   W = 180.000 deg,  P = 45.000 deg,  R = 180.000 deg
};
/END
