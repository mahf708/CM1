! bind(C) shims around the MM5/WRF SLAB land model (slab.F: SLAB/SLAB1D),
! as turb.F (sfc_and_turb, line 1021) calls it, plus a verbatim copy of the
! soil-layer setup of init_surface.F (slab_zs, slab_dzs, initial tslb) and a
! wrapper of getlanduse (reads ./LANDUSE.TBL). None of these read the
! `input` module: dimensions come as arguments.
!
! The 2D arrays travel packed as s2d(ib:ie,jb:je,18), in this order (must
! match slab_tests.cpp):
!   1 flhc  2 flqc  3 psfc  4 xland  5 tmn  6 hfx  7 qfx  8 lh  9 tsk
!  10 qsfc 11 chklowq 12 gsw 13 glw 14 capg 15 thc 16 snowc 17 emiss
!  18 mavail
! and the constants as rc(19) (cm1f_bridge_sfclay.F90's order):
!   cp g rovcp rd xlv lv1 lv2 svp1 svp2 svp3 svpt0 ep1 ep2 karman eomeg
!   stbolt p1000mb rv rovg

! turb.F 1021: call SLAB(dum4,divx,prs,FLHC,FLQC,PSFC,XLAND,TMN,HFX,QFX,LH,
! TSK,QSFC,CHKLOWQ,GSW,GLW,CAPG,THC,SNOWC,EMISS,MAVAIL,DT,ROVCP,XLV,lv1,lv2,
! DTMIN,IFSNOW,SVP1,SVP2,SVP3,SVPT0,EP2,KARMAN,EOMEG,STBOLT,TSLB,slab_ZS,
! slab_DZS,num_soil_layers,.true.,P1000mb, ...)
subroutine cm1f_slab(ni, nj, nk, ib, ie, jb, je, kb, ke, nsl, ifsnow, irad, deltsm, rc, zs, dzs, &
                     t3d, qv3d, p3d, s2d, tslb) bind(C, name="cm1f_slab")
  use iso_c_binding
  use module_sf_slab, only: slab
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke, nsl, ifsnow, irad
  real(c_double), value :: deltsm
  real(c_double), intent(in) :: rc(19), zs(nsl), dzs(nsl)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: t3d, qv3d, p3d
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,18), tslb(ib:ie,jb:je,nsl)
  real :: ROVCP, XLV, lv1, lv2, SVP1, SVP2, SVP3, SVPT0, EP2, KARMAN, EOMEG, STBOLT, P1000mb, DTMIN
  logical :: radiation
  ROVCP = rc(3); XLV = rc(5); lv1 = rc(6); lv2 = rc(7)
  SVP1 = rc(8); SVP2 = rc(9); SVP3 = rc(10); SVPT0 = rc(11); EP2 = rc(13)
  KARMAN = rc(14); EOMEG = rc(15); STBOLT = rc(16); P1000mb = rc(17)
  DTMIN = deltsm/60.0
  radiation = (irad .ne. 0)
  call SLAB(t3d,qv3d,p3d,s2d(ib,jb,1),s2d(ib,jb,2),                                   &
               s2d(ib,jb,3),s2d(ib,jb,4),s2d(ib,jb,5),s2d(ib,jb,6),s2d(ib,jb,7),       &
               s2d(ib,jb,8),s2d(ib,jb,9),s2d(ib,jb,10),s2d(ib,jb,11),                  &
               s2d(ib,jb,12),s2d(ib,jb,13),s2d(ib,jb,14),s2d(ib,jb,15),s2d(ib,jb,16),  &
               s2d(ib,jb,17),s2d(ib,jb,18),                                            &
               deltsm,ROVCP,XLV,lv1,lv2,DTMIN,ifsnow,                                  &
               SVP1,SVP2,SVP3,SVPT0,EP2,                                               &
               KARMAN,EOMEG,STBOLT,                                                    &
               tslb,zs,dzs,nsl,radiation,                                              &
               P1000mb,                                                                &
                 1, ni+1,   1, nj+1,   1, nk+1,                                        &
                ib, ie,  jb, je,  kb, ke,                                              &
                 1, ni,   1, nj,   1, nk                    )
end subroutine cm1f_slab

! init_surface.F (sfcmodel = 2,3,4,6,7), verbatim: soil layers and the
! initial soil temperatures
subroutine cm1f_slab_soil(num_soil_layers, ib, ie, jb, je, tsk, tmn, slab_zs, slab_dzs, tslb) &
    bind(C, name="cm1f_slab_soil")
  use iso_c_binding
  implicit none
  integer(c_int), value :: num_soil_layers, ib, ie, jb, je
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: tsk, tmn
  real(c_double), intent(inout) :: slab_zs(num_soil_layers), slab_dzs(num_soil_layers)
  real(c_double), intent(inout) :: tslb(ib:ie,jb:je,num_soil_layers)
  integer :: i, j, l
      IF (num_soil_layers.NE.1)THEN
      slab_dzs(1)=.01
      slab_zs(1)=.5*slab_dzs(1)
      DO l=2,num_soil_layers
         slab_dzs(l)=2*slab_dzs(l-1)
         slab_zs(l)=slab_zs(l-1)+.5*slab_dzs(l-1)+.5*slab_dzs(l)
      ENDDO
      DO l=1,num_soil_layers
         DO j=jb,je
            DO i=ib,ie
              tslb(i,j,l)=( tsk(i,j)*(slab_zs(num_soil_layers)-slab_zs(l))   &
                          + tmn(i,j)*(slab_zs(l)-slab_zs(1)) ) / &
                          ( slab_zs(num_soil_layers)-slab_zs(1) )
            ENDDO
         ENDDO
      ENDDO
      ENDIF
end subroutine cm1f_slab_soil

! init_surface.F getlanduse for one point of category lu (./LANDUSE.TBL):
! emissivity and thermal inertia (SLAB's emiss, thc)
subroutine cm1f_getlanduse(season, lu, emiss_out, thc_out) bind(C, name="cm1f_getlanduse")
  use iso_c_binding
  use init_surface_module, only: getlanduse
  implicit none
  integer(c_int), value :: season, lu
  real(c_double), intent(out) :: emiss_out, thc_out
  integer :: lu_index(1,1)
  real :: emiss(1,1), thc(1,1), albedo(1,1), mavail(1,1), znt(1,1), xland(1,1)
  lu_index = lu
  emiss = 0; thc = 0; albedo = 0; mavail = 0; znt = 0; xland = 0
  call getlanduse(season, 0, 1, 1, 1, 1, 1, 1, 1, 1, lu_index, xland, emiss, thc, albedo, znt, mavail)
  emiss_out = emiss(1,1)
  thc_out = thc(1,1)
end subroutine cm1f_getlanduse
