! bind(C) shims for the RRTMG_LW tables (module_ra_rrtmg_lw.F) after
! rrtmg_lwinit: the C++ port's tables are compared array by array.
! Generated once from cm1k_rrtmg_lw_tables.cpp's layout (names match).

subroutine cm1f_rrtmg_lwinit(p_top, kme, nlayers_out) bind(C, name="cm1f_rrtmg_lwinit")
  use iso_c_binding
  use module_ra_rrtmg_lw, only: rrtmg_lwinit, nlayers
  implicit none
  real(c_double), value :: p_top
  integer(c_int), value :: kme
  integer(c_int), intent(out) :: nlayers_out
  call rrtmg_lwinit(p_top, .true., 0, 1, 2, 1, 2, 1, kme, 1, 1, 1, 1, 1, kme, 1, 1, 1, 1, 1, kme - 1)
  nlayers_out = nlayers
end subroutine cm1f_rrtmg_lwinit

! Scalars: fluxfac, heatfac, oneminus, pi, grav, planck, boltz, clight,
! avogad, alosmt, gascon, radcn1, radcn2, sbcnst, secdy, abscld1, absliq0,
! bpade, wtdiff, rec_6, nxmol.
subroutine cm1f_rrtmg_lw_scalars(v) bind(C, name="cm1f_rrtmg_lw_scalars")
  use iso_c_binding
  use rrlw_con
  use rrlw_cld, only: abscld1, absliq0
  use rrlw_tbl, only: bpade
  use rrlw_wvn, only: nxmol
  use rrtmg_lw_rtrnmc, only: wtdiff, rec_6
  implicit none
  real(c_double), intent(out) :: v(21)
  v = (/ fluxfac, heatfac, oneminus, pi, grav, planck, boltz, clight, avogad, alosmt, gascon, radcn1, radcn2, &
         sbcnst, secdy, abscld1, absliq0, bpade, wtdiff, rec_6, real(nxmol, c_double) /)
end subroutine cm1f_rrtmg_lw_scalars

! Copies table `name` (as named in the C++ registry) to out(n); returns the
! element count it has, or -1 if the name is unknown.
function cm1f_rrtmg_lw_table(cname, out, n) result(cnt) bind(C, name="cm1f_rrtmg_lw_table")
  use iso_c_binding
  use rrlw_cld, only: t_absice0 => absice0
  use rrlw_cld, only: t_absice1 => absice1
  use rrlw_cld, only: t_absice2 => absice2
  use rrlw_cld, only: t_absice3 => absice3
  use rrlw_cld, only: t_absliq1 => absliq1
  use rrlw_ref, only: t_pref => pref
  use rrlw_ref, only: t_preflog => preflog
  use rrlw_ref, only: t_tref => tref
  use rrlw_ref, only: t_chi_mls => chi_mls
  use rrlw_tbl, only: t_tau_tbl => tau_tbl
  use rrlw_tbl, only: t_exp_tbl => exp_tbl
  use rrlw_tbl, only: t_tfn_tbl => tfn_tbl
  use rrlw_wvn, only: t_ng => ng
  use rrlw_wvn, only: t_nspa => nspa
  use rrlw_wvn, only: t_nspb => nspb
  use rrlw_wvn, only: t_wavenum1 => wavenum1
  use rrlw_wvn, only: t_wavenum2 => wavenum2
  use rrlw_wvn, only: t_delwave => delwave
  use rrlw_wvn, only: t_totplnk => totplnk
  use rrlw_wvn, only: t_totplk16 => totplk16
  use rrlw_wvn, only: t_ngc => ngc
  use rrlw_wvn, only: t_ngs => ngs
  use rrlw_wvn, only: t_ngn => ngn
  use rrlw_wvn, only: t_ngb => ngb
  use rrlw_wvn, only: t_ngm => ngm
  use rrlw_wvn, only: t_wt => wt
  use rrlw_wvn, only: t_rwgt => rwgt
  use rrlw_wvn, only: t_ixindx => ixindx
  use rrtmg_lw_rtrnmc, only: t_a0 => a0
  use rrtmg_lw_rtrnmc, only: t_a1 => a1
  use rrtmg_lw_rtrnmc, only: t_a2 => a2
  use rrlw_kg01, only: kg01_fracrefa => fracrefa
  use rrlw_kg01, only: kg01_fracrefb => fracrefb
  use rrlw_kg01, only: kg01_absa => absa
  use rrlw_kg01, only: kg01_absb => absb
  use rrlw_kg01, only: kg01_ka_mn2 => ka_mn2
  use rrlw_kg01, only: kg01_kb_mn2 => kb_mn2
  use rrlw_kg01, only: kg01_selfref => selfref
  use rrlw_kg01, only: kg01_forref => forref
  use rrlw_kg02, only: kg02_fracrefa => fracrefa
  use rrlw_kg02, only: kg02_fracrefb => fracrefb
  use rrlw_kg02, only: kg02_absa => absa
  use rrlw_kg02, only: kg02_absb => absb
  use rrlw_kg02, only: kg02_selfref => selfref
  use rrlw_kg02, only: kg02_forref => forref
  use rrlw_kg03, only: kg03_fracrefa => fracrefa
  use rrlw_kg03, only: kg03_fracrefb => fracrefb
  use rrlw_kg03, only: kg03_absa => absa
  use rrlw_kg03, only: kg03_absb => absb
  use rrlw_kg03, only: kg03_ka_mn2o => ka_mn2o
  use rrlw_kg03, only: kg03_kb_mn2o => kb_mn2o
  use rrlw_kg03, only: kg03_selfref => selfref
  use rrlw_kg03, only: kg03_forref => forref
  use rrlw_kg04, only: kg04_fracrefa => fracrefa
  use rrlw_kg04, only: kg04_fracrefb => fracrefb
  use rrlw_kg04, only: kg04_absa => absa
  use rrlw_kg04, only: kg04_absb => absb
  use rrlw_kg04, only: kg04_selfref => selfref
  use rrlw_kg04, only: kg04_forref => forref
  use rrlw_kg05, only: kg05_fracrefa => fracrefa
  use rrlw_kg05, only: kg05_fracrefb => fracrefb
  use rrlw_kg05, only: kg05_absa => absa
  use rrlw_kg05, only: kg05_absb => absb
  use rrlw_kg05, only: kg05_ka_mo3 => ka_mo3
  use rrlw_kg05, only: kg05_ccl4 => ccl4
  use rrlw_kg05, only: kg05_selfref => selfref
  use rrlw_kg05, only: kg05_forref => forref
  use rrlw_kg06, only: kg06_fracrefa => fracrefa
  use rrlw_kg06, only: kg06_absa => absa
  use rrlw_kg06, only: kg06_ka_mco2 => ka_mco2
  use rrlw_kg06, only: kg06_cfc11adj => cfc11adj
  use rrlw_kg06, only: kg06_cfc12 => cfc12
  use rrlw_kg06, only: kg06_selfref => selfref
  use rrlw_kg06, only: kg06_forref => forref
  use rrlw_kg07, only: kg07_fracrefa => fracrefa
  use rrlw_kg07, only: kg07_fracrefb => fracrefb
  use rrlw_kg07, only: kg07_absa => absa
  use rrlw_kg07, only: kg07_absb => absb
  use rrlw_kg07, only: kg07_ka_mco2 => ka_mco2
  use rrlw_kg07, only: kg07_kb_mco2 => kb_mco2
  use rrlw_kg07, only: kg07_selfref => selfref
  use rrlw_kg07, only: kg07_forref => forref
  use rrlw_kg08, only: kg08_fracrefa => fracrefa
  use rrlw_kg08, only: kg08_fracrefb => fracrefb
  use rrlw_kg08, only: kg08_absa => absa
  use rrlw_kg08, only: kg08_absb => absb
  use rrlw_kg08, only: kg08_ka_mco2 => ka_mco2
  use rrlw_kg08, only: kg08_kb_mco2 => kb_mco2
  use rrlw_kg08, only: kg08_ka_mn2o => ka_mn2o
  use rrlw_kg08, only: kg08_kb_mn2o => kb_mn2o
  use rrlw_kg08, only: kg08_ka_mo3 => ka_mo3
  use rrlw_kg08, only: kg08_cfc12 => cfc12
  use rrlw_kg08, only: kg08_cfc22adj => cfc22adj
  use rrlw_kg08, only: kg08_selfref => selfref
  use rrlw_kg08, only: kg08_forref => forref
  use rrlw_kg09, only: kg09_fracrefa => fracrefa
  use rrlw_kg09, only: kg09_fracrefb => fracrefb
  use rrlw_kg09, only: kg09_absa => absa
  use rrlw_kg09, only: kg09_absb => absb
  use rrlw_kg09, only: kg09_ka_mn2o => ka_mn2o
  use rrlw_kg09, only: kg09_kb_mn2o => kb_mn2o
  use rrlw_kg09, only: kg09_selfref => selfref
  use rrlw_kg09, only: kg09_forref => forref
  use rrlw_kg10, only: kg10_fracrefa => fracrefa
  use rrlw_kg10, only: kg10_fracrefb => fracrefb
  use rrlw_kg10, only: kg10_absa => absa
  use rrlw_kg10, only: kg10_absb => absb
  use rrlw_kg10, only: kg10_selfref => selfref
  use rrlw_kg10, only: kg10_forref => forref
  use rrlw_kg11, only: kg11_fracrefa => fracrefa
  use rrlw_kg11, only: kg11_fracrefb => fracrefb
  use rrlw_kg11, only: kg11_absa => absa
  use rrlw_kg11, only: kg11_absb => absb
  use rrlw_kg11, only: kg11_ka_mo2 => ka_mo2
  use rrlw_kg11, only: kg11_kb_mo2 => kb_mo2
  use rrlw_kg11, only: kg11_selfref => selfref
  use rrlw_kg11, only: kg11_forref => forref
  use rrlw_kg12, only: kg12_fracrefa => fracrefa
  use rrlw_kg12, only: kg12_absa => absa
  use rrlw_kg12, only: kg12_selfref => selfref
  use rrlw_kg12, only: kg12_forref => forref
  use rrlw_kg13, only: kg13_fracrefa => fracrefa
  use rrlw_kg13, only: kg13_fracrefb => fracrefb
  use rrlw_kg13, only: kg13_absa => absa
  use rrlw_kg13, only: kg13_ka_mco2 => ka_mco2
  use rrlw_kg13, only: kg13_ka_mco => ka_mco
  use rrlw_kg13, only: kg13_kb_mo3 => kb_mo3
  use rrlw_kg13, only: kg13_selfref => selfref
  use rrlw_kg13, only: kg13_forref => forref
  use rrlw_kg14, only: kg14_fracrefa => fracrefa
  use rrlw_kg14, only: kg14_fracrefb => fracrefb
  use rrlw_kg14, only: kg14_absa => absa
  use rrlw_kg14, only: kg14_absb => absb
  use rrlw_kg14, only: kg14_selfref => selfref
  use rrlw_kg14, only: kg14_forref => forref
  use rrlw_kg15, only: kg15_fracrefa => fracrefa
  use rrlw_kg15, only: kg15_absa => absa
  use rrlw_kg15, only: kg15_ka_mn2 => ka_mn2
  use rrlw_kg15, only: kg15_selfref => selfref
  use rrlw_kg15, only: kg15_forref => forref
  use rrlw_kg16, only: kg16_fracrefa => fracrefa
  use rrlw_kg16, only: kg16_fracrefb => fracrefb
  use rrlw_kg16, only: kg16_absa => absa
  use rrlw_kg16, only: kg16_absb => absb
  use rrlw_kg16, only: kg16_selfref => selfref
  use rrlw_kg16, only: kg16_forref => forref
  implicit none
  character(kind=c_char), intent(in) :: cname(*)
  integer(c_int), value :: n
  real(c_double), intent(out) :: out(n)
  integer(c_int) :: cnt
  character(len=64) :: name
  integer :: i
  name = ' '
  do i = 1, 64
    if (cname(i) == c_null_char) exit
    name(i:i) = cname(i)
  enddo
  cnt = -1
  select case (trim(name))
  case ('absice0'); call put_r(t_absice0)
  case ('absice1'); call put_r(t_absice1)
  case ('absice2'); call put_r(t_absice2)
  case ('absice3'); call put_r(t_absice3)
  case ('absliq1'); call put_r(t_absliq1)
  case ('pref'); call put_r(t_pref)
  case ('preflog'); call put_r(t_preflog)
  case ('tref'); call put_r(t_tref)
  case ('chi_mls'); call put_r(t_chi_mls)
  case ('tau_tbl'); call put_r(t_tau_tbl)
  case ('exp_tbl'); call put_r(t_exp_tbl)
  case ('tfn_tbl'); call put_r(t_tfn_tbl)
  case ('wavenum1'); call put_r(t_wavenum1)
  case ('wavenum2'); call put_r(t_wavenum2)
  case ('delwave'); call put_r(t_delwave)
  case ('totplnk'); call put_r(t_totplnk)
  case ('totplk16'); call put_r(t_totplk16)
  case ('wt'); call put_r(t_wt)
  case ('rwgt'); call put_r(t_rwgt)
  case ('a0'); call put_r(t_a0)
  case ('a1'); call put_r(t_a1)
  case ('a2'); call put_r(t_a2)
  case ('kg01.fracrefa'); call put_r(kg01_fracrefa)
  case ('kg01.fracrefb'); call put_r(kg01_fracrefb)
  case ('kg01.absa'); call put_r(kg01_absa)
  case ('kg01.absb'); call put_r(kg01_absb)
  case ('kg01.ka_mn2'); call put_r(kg01_ka_mn2)
  case ('kg01.kb_mn2'); call put_r(kg01_kb_mn2)
  case ('kg01.selfref'); call put_r(kg01_selfref)
  case ('kg01.forref'); call put_r(kg01_forref)
  case ('kg02.fracrefa'); call put_r(kg02_fracrefa)
  case ('kg02.fracrefb'); call put_r(kg02_fracrefb)
  case ('kg02.absa'); call put_r(kg02_absa)
  case ('kg02.absb'); call put_r(kg02_absb)
  case ('kg02.selfref'); call put_r(kg02_selfref)
  case ('kg02.forref'); call put_r(kg02_forref)
  case ('kg03.fracrefa'); call put_r(kg03_fracrefa)
  case ('kg03.fracrefb'); call put_r(kg03_fracrefb)
  case ('kg03.absa'); call put_r(kg03_absa)
  case ('kg03.absb'); call put_r(kg03_absb)
  case ('kg03.ka_mn2o'); call put_r(kg03_ka_mn2o)
  case ('kg03.kb_mn2o'); call put_r(kg03_kb_mn2o)
  case ('kg03.selfref'); call put_r(kg03_selfref)
  case ('kg03.forref'); call put_r(kg03_forref)
  case ('kg04.fracrefa'); call put_r(kg04_fracrefa)
  case ('kg04.fracrefb'); call put_r(kg04_fracrefb)
  case ('kg04.absa'); call put_r(kg04_absa)
  case ('kg04.absb'); call put_r(kg04_absb)
  case ('kg04.selfref'); call put_r(kg04_selfref)
  case ('kg04.forref'); call put_r(kg04_forref)
  case ('kg05.fracrefa'); call put_r(kg05_fracrefa)
  case ('kg05.fracrefb'); call put_r(kg05_fracrefb)
  case ('kg05.absa'); call put_r(kg05_absa)
  case ('kg05.absb'); call put_r(kg05_absb)
  case ('kg05.ka_mo3'); call put_r(kg05_ka_mo3)
  case ('kg05.ccl4'); call put_r(kg05_ccl4)
  case ('kg05.selfref'); call put_r(kg05_selfref)
  case ('kg05.forref'); call put_r(kg05_forref)
  case ('kg06.fracrefa'); call put_r(kg06_fracrefa)
  case ('kg06.absa'); call put_r(kg06_absa)
  case ('kg06.ka_mco2'); call put_r(kg06_ka_mco2)
  case ('kg06.cfc11adj'); call put_r(kg06_cfc11adj)
  case ('kg06.cfc12'); call put_r(kg06_cfc12)
  case ('kg06.selfref'); call put_r(kg06_selfref)
  case ('kg06.forref'); call put_r(kg06_forref)
  case ('kg07.fracrefa'); call put_r(kg07_fracrefa)
  case ('kg07.fracrefb'); call put_r(kg07_fracrefb)
  case ('kg07.absa'); call put_r(kg07_absa)
  case ('kg07.absb'); call put_r(kg07_absb)
  case ('kg07.ka_mco2'); call put_r(kg07_ka_mco2)
  case ('kg07.kb_mco2'); call put_r(kg07_kb_mco2)
  case ('kg07.selfref'); call put_r(kg07_selfref)
  case ('kg07.forref'); call put_r(kg07_forref)
  case ('kg08.fracrefa'); call put_r(kg08_fracrefa)
  case ('kg08.fracrefb'); call put_r(kg08_fracrefb)
  case ('kg08.absa'); call put_r(kg08_absa)
  case ('kg08.absb'); call put_r(kg08_absb)
  case ('kg08.ka_mco2'); call put_r(kg08_ka_mco2)
  case ('kg08.kb_mco2'); call put_r(kg08_kb_mco2)
  case ('kg08.ka_mn2o'); call put_r(kg08_ka_mn2o)
  case ('kg08.kb_mn2o'); call put_r(kg08_kb_mn2o)
  case ('kg08.ka_mo3'); call put_r(kg08_ka_mo3)
  case ('kg08.cfc12'); call put_r(kg08_cfc12)
  case ('kg08.cfc22adj'); call put_r(kg08_cfc22adj)
  case ('kg08.selfref'); call put_r(kg08_selfref)
  case ('kg08.forref'); call put_r(kg08_forref)
  case ('kg09.fracrefa'); call put_r(kg09_fracrefa)
  case ('kg09.fracrefb'); call put_r(kg09_fracrefb)
  case ('kg09.absa'); call put_r(kg09_absa)
  case ('kg09.absb'); call put_r(kg09_absb)
  case ('kg09.ka_mn2o'); call put_r(kg09_ka_mn2o)
  case ('kg09.kb_mn2o'); call put_r(kg09_kb_mn2o)
  case ('kg09.selfref'); call put_r(kg09_selfref)
  case ('kg09.forref'); call put_r(kg09_forref)
  case ('kg10.fracrefa'); call put_r(kg10_fracrefa)
  case ('kg10.fracrefb'); call put_r(kg10_fracrefb)
  case ('kg10.absa'); call put_r(kg10_absa)
  case ('kg10.absb'); call put_r(kg10_absb)
  case ('kg10.selfref'); call put_r(kg10_selfref)
  case ('kg10.forref'); call put_r(kg10_forref)
  case ('kg11.fracrefa'); call put_r(kg11_fracrefa)
  case ('kg11.fracrefb'); call put_r(kg11_fracrefb)
  case ('kg11.absa'); call put_r(kg11_absa)
  case ('kg11.absb'); call put_r(kg11_absb)
  case ('kg11.ka_mo2'); call put_r(kg11_ka_mo2)
  case ('kg11.kb_mo2'); call put_r(kg11_kb_mo2)
  case ('kg11.selfref'); call put_r(kg11_selfref)
  case ('kg11.forref'); call put_r(kg11_forref)
  case ('kg12.fracrefa'); call put_r(kg12_fracrefa)
  case ('kg12.absa'); call put_r(kg12_absa)
  case ('kg12.selfref'); call put_r(kg12_selfref)
  case ('kg12.forref'); call put_r(kg12_forref)
  case ('kg13.fracrefa'); call put_r(kg13_fracrefa)
  case ('kg13.fracrefb'); call put_r(kg13_fracrefb)
  case ('kg13.absa'); call put_r(kg13_absa)
  case ('kg13.ka_mco2'); call put_r(kg13_ka_mco2)
  case ('kg13.ka_mco'); call put_r(kg13_ka_mco)
  case ('kg13.kb_mo3'); call put_r(kg13_kb_mo3)
  case ('kg13.selfref'); call put_r(kg13_selfref)
  case ('kg13.forref'); call put_r(kg13_forref)
  case ('kg14.fracrefa'); call put_r(kg14_fracrefa)
  case ('kg14.fracrefb'); call put_r(kg14_fracrefb)
  case ('kg14.absa'); call put_r(kg14_absa)
  case ('kg14.absb'); call put_r(kg14_absb)
  case ('kg14.selfref'); call put_r(kg14_selfref)
  case ('kg14.forref'); call put_r(kg14_forref)
  case ('kg15.fracrefa'); call put_r(kg15_fracrefa)
  case ('kg15.absa'); call put_r(kg15_absa)
  case ('kg15.ka_mn2'); call put_r(kg15_ka_mn2)
  case ('kg15.selfref'); call put_r(kg15_selfref)
  case ('kg15.forref'); call put_r(kg15_forref)
  case ('kg16.fracrefa'); call put_r(kg16_fracrefa)
  case ('kg16.fracrefb'); call put_r(kg16_fracrefb)
  case ('kg16.absa'); call put_r(kg16_absa)
  case ('kg16.absb'); call put_r(kg16_absb)
  case ('kg16.selfref'); call put_r(kg16_selfref)
  case ('kg16.forref'); call put_r(kg16_forref)
  case ('ng'); call put_i(t_ng)
  case ('nspa'); call put_i(t_nspa)
  case ('nspb'); call put_i(t_nspb)
  case ('ngc'); call put_i(t_ngc)
  case ('ngs'); call put_i(t_ngs)
  case ('ngn'); call put_i(t_ngn)
  case ('ngb'); call put_i(t_ngb)
  case ('ngm'); call put_i(t_ngm)
  case ('ixindx'); call put_i(t_ixindx)
  end select
contains
  subroutine put_r(a)
    real(kind=8), intent(in) :: a(*)
    integer :: k
    do k = 1, n
      out(k) = a(k)
    enddo
    cnt = n
  end subroutine put_r
  subroutine put_i(a)
    integer, intent(in) :: a(*)
    integer :: k
    do k = 1, n
      out(k) = real(a(k), c_double)
    enddo
    cnt = n
  end subroutine put_i
end function cm1f_rrtmg_lw_table
