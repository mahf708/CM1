! bind(C) shims around RRTMG_LWRAD / RRTMG_SWRAD (module_ra_rrtmg_lw.F,
! module_ra_rrtmg_sw.F), called with the arguments radiation_driver.F passes
! for radopt = 2 (has_req* = 0, o3input = 0, icloud = 1, no aerosols).
! 3-D arrays are (ib:ie,jb:je,kb:ke) with ib = 1-ngxy, kb = 0, ke = nk+1;
! 2-D arrays (ib:ie,jb:je).

subroutine cm1f_rrtmg_lwrad(ni, nj, nk, ngxy, cldovrlp, yr, julian, r, g, fq, &
                            p8w, p3d, pi3d, dz8w, t3d, t8w, rho3d, cldfra, qv, qc, qr, qi, qs, qg, o30, &
                            tsk, emiss, xland, xice, xsnow, &
                            lwten, lwtenc, glw, olr, lwcf, lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, &
                            lwdnbc, co2out) bind(C, name="cm1f_rrtmg_lwrad")
  use iso_c_binding
  use module_ra_rrtmg_lw, only: rrtmg_lwrad
  implicit none
  integer(c_int), value :: ni, nj, nk, ngxy, cldovrlp, yr
  real(c_double), value :: julian, r, g
  integer(c_int), intent(in) :: fq(5)     ! f_qc, f_qr, f_qi, f_qs, f_qg
  real(c_double), dimension(1-ngxy:ni+ngxy, 1-ngxy:nj+ngxy, 0:nk+1) :: p8w, p3d, pi3d, dz8w, t3d, t8w, rho3d, &
      cldfra, qv, qc, qr, qi, qs, qg, o30, lwten, lwtenc
  real(c_double), dimension(1-ngxy:ni+ngxy, 1-ngxy:nj+ngxy) :: tsk, emiss, xland, xice, xsnow, glw, olr, lwcf, &
      lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc
  real(c_double), intent(out) :: co2out
  logical :: f_qv, f_qc, f_qr, f_qi, f_qs, f_qg
  integer :: ib, ie, jb, je, kb, ke
  ib = 1-ngxy ; ie = ni+ngxy ; jb = 1-ngxy ; je = nj+ngxy ; kb = 0 ; ke = nk+1
  f_qv = .true.
  f_qc = fq(1) /= 0 ; f_qr = fq(2) /= 0 ; f_qi = fq(3) /= 0 ; f_qs = fq(4) /= 0 ; f_qg = fq(5) /= 0
  CALL RRTMG_LWRAD(                                      &
       RTHRATENLW=lwten,lwtenc=lwtenc,                   &
       LWUPT=LWUPT,LWUPTC=LWUPTC,                        &
       LWDNT=LWDNT,LWDNTC=LWDNTC,                        &
       LWUPB=LWUPB,LWUPBC=LWUPBC,                        &
       LWDNB=LWDNB,LWDNBC=LWDNBC,                        &
       GLW=GLW,OLR=olr,LWCF=LWCF,                        &
       EMISS=EMISS,                                      &
       P8W=p8w,P3D=p3d,PI3D=pi3d,DZ8W=dz8w,TSK=tsk,T3D=t3d,    &
       T8W=t8w,RHO3D=rho3d,R=r,G=g,                      &
       ICLOUD=1,WARM_RAIN=.false.,                       &
       CLDFRA3D=CLDFRA,                                  &
       cldovrlp=cldovrlp,                                &
       IS_CAMMGMP_USED=.false.,                          &
       XLAND=XLAND,XICE=XICE,SNOW=XSNOW,                 &
       QV3D=qv,QC3D=qc,QR3D=qr,QI3D=qi,QS3D=qs,QG3D=qg,  &
       O3INPUT=0,O33D=o30,                               &
       F_QV=F_QV,F_QC=F_QC,F_QR=F_QR,                    &
       F_QI=F_QI,F_QS=F_QS,F_QG=F_QG,                    &
       RE_CLOUD=qg,RE_ICE=qg,RE_SNOW=qg,              &
       has_reqc=0,has_reqi=0,has_reqs=0,                 &
       YR=YR,JULIAN=JULIAN,                              &
       calc_clean_atm_diag=0,                            &
       ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=nk+1 ,          &
       ims=ib ,ime=ie   , jms=jb ,jme=je   , kms=kb ,kme=ke ,            &
       its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=nk ,            &
       mp_physics=1,co2out=co2out               )
end subroutine cm1f_rrtmg_lwrad

subroutine cm1f_rrtmg_swrad(ni, nj, nk, ngxy, cldovrlp, yr, julian, solcon, r, g, fq, &
                            p8w, p3d, pi3d, dz8w, t3d, t8w, rho3d, cldfra, qv, qc, qr, qi, qs, qg, o30, &
                            tsk, albd, xland, xice, xsnow, xlat, xlong, coszen, &
                            swten, swtenc, gsw, swcf, coszr, swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, &
                            swdnbc, swddir, swddni, swddif, swdownc, swddnic, swddirc, co2out) &
                            bind(C, name="cm1f_rrtmg_swrad")
  use iso_c_binding
  use module_ra_rrtmg_sw, only: rrtmg_swrad
  implicit none
  integer(c_int), value :: ni, nj, nk, ngxy, cldovrlp, yr
  real(c_double), value :: julian, solcon, r, g
  integer(c_int), intent(in) :: fq(5)     ! f_qc, f_qr, f_qi, f_qs, f_qg
  real(c_double), dimension(1-ngxy:ni+ngxy, 1-ngxy:nj+ngxy, 0:nk+1) :: p8w, p3d, pi3d, dz8w, t3d, t8w, rho3d, &
      cldfra, qv, qc, qr, qi, qs, qg, o30, swten, swtenc
  real(c_double), dimension(1-ngxy:ni+ngxy, 1-ngxy:nj+ngxy) :: tsk, albd, xland, xice, xsnow, xlat, xlong, &
      coszen, gsw, swcf, coszr, swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc, &
      swddir, swddni, swddif, swdownc, swddnic, swddirc
  real(c_double), intent(out) :: co2out
  real, dimension(:,:,:,:), pointer :: tauaer_sw=>null(), ssaaer_sw=>null(), asyaer_sw=>null()
  logical :: f_qv, f_qc, f_qr, f_qi, f_qs, f_qg
  integer :: ib, ie, jb, je, kb, ke, aer_opt
  real :: xtime, gmt, radt_min, degrad, declin
  ib = 1-ngxy ; ie = ni+ngxy ; jb = 1-ngxy ; je = nj+ngxy ; kb = 0 ; ke = nk+1
  f_qv = .true.
  f_qc = fq(1) /= 0 ; f_qr = fq(2) /= 0 ; f_qi = fq(3) /= 0 ; f_qs = fq(4) /= 0 ; f_qg = fq(5) /= 0
  ! RRTMG_SWRAD uses xcoszen, not these
  xtime = 0.0 ; gmt = 0.0 ; radt_min = 1.0 ; degrad = 3.1415926/180. ; declin = 0.0
  aer_opt = 0
  CALL RRTMG_SWRAD(                                         &
       RTHRATENSW=swten,swtenc=swtenc,                   &
       SWUPT=SWUPT,SWUPTC=SWUPTC,                        &
       SWDNT=SWDNT,SWDNTC=SWDNTC,                        &
       SWUPB=SWUPB,SWUPBC=SWUPBC,                        &
       SWDNB=SWDNB,SWDNBC=SWDNBC,                        &
       SWCF=SWCF,GSW=GSW,                                &
       XTIME=XTIME,GMT=GMT,XLAT=XLAT,XLONG=XLONG,        &
       RADT=radt_min,DEGRAD=DEGRAD,DECLIN=DECLIN,        &
       COSZR=COSZR,JULDAY=int(julian),SOLCON=SOLCON,     &
       ALBEDO=ALBD,t3d=t3d,t8w=t8w,TSK=TSK,              &
       p3d=p3d,p8w=p8w,pi3d=pi3d,rho3d=rho3d,            &
       dz8w=dz8w,                                        &
       CLDFRA3D=CLDFRA,                                  &
       IS_CAMMGMP_USED=.false.,                          &
       R=r,G=G,                                          &
       ICLOUD=1,WARM_RAIN=.false.,                       &
       cldovrlp=cldovrlp,                                &
       XLAND=XLAND,XICE=XICE,SNOW=XSNOW,                 &
       QV3D=qv,QC3D=qc,QR3D=qr,QI3D=qi,QS3D=qs,QG3D=qg,  &
       O3INPUT=0,O33D=o30,                               &
       AER_OPT=AER_OPT,                                  &
       no_src=6,                                         &
       SF_SURFACE_PHYSICS=1,                             &
       F_QV=f_qv,F_QC=f_qc,F_QR=f_qr,                    &
       F_QI=f_qi,F_QS=f_qs,F_QG=f_qg,                    &
       RE_CLOUD=qg,RE_ICE=qg,RE_SNOW=qg,                 &
       has_reqc=0,has_reqi=0,has_reqs=0,                 &
       calc_clean_atm_diag=0,                            &
       ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=nk+1 ,          &
       ims=ib ,ime=ie   , jms=jb ,jme=je   , kms=kb ,kme=ke ,            &
       its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=nk ,            &
       tauaer3d_sw=tauaer_sw,                            &
       ssaaer3d_sw=ssaaer_sw,                            &
       asyaer3d_sw=asyaer_sw,                            &
       swddir=swddir,swddni=swddni,swddif=swddif,        &
       swdownc=swdownc, swddnic=swddnic, swddirc=swddirc, &
       xcoszen=coszen,yr=yr,julian=julian,mp_physics=1,co2out=co2out )
end subroutine cm1f_rrtmg_swrad
