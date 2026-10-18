! bind(C) shims for the RRTMG_SW tables (module_ra_rrtmg_sw.F) after
! rrtmg_swinit; generated once from cm1k_rrtmg_sw_tables.cpp's layout.

subroutine cm1f_rrtmg_swinit() bind(C, name="cm1f_rrtmg_swinit")
  use iso_c_binding
  use module_ra_rrtmg_sw, only: rrtmg_swinit
  implicit none
  call rrtmg_swinit(.true., 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1)
end subroutine cm1f_rrtmg_swinit

! heatfac, oneminus, pi, grav, planck, boltz, clight, avogad, alosmt,
! gascon, radcn1, radcn2, sbcnst, secdy, bpade, tau_tbl
subroutine cm1f_rrtmg_sw_scalars(v) bind(C, name="cm1f_rrtmg_sw_scalars")
  use iso_c_binding
  use rrsw_con
  use rrsw_tbl, only: bpade, tau_tbl
  implicit none
  real(c_double), intent(out) :: v(16)
  v = (/ heatfac, oneminus, pi, grav, planck, boltz, clight, avogad, alosmt, gascon, radcn1, radcn2, &
         sbcnst, secdy, bpade, tau_tbl /)
end subroutine cm1f_rrtmg_sw_scalars

! band scalar "kgNN.name" (rayl, strrat, strrat1, givfac, scalekur, layreffr)
function cm1f_rrtmg_sw_scalar(cname) result(v) bind(C, name="cm1f_rrtmg_sw_scalar")
  use iso_c_binding
  use rrsw_kg16, only: q16_rayl => rayl
  use rrsw_kg16, only: q16_strrat1 => strrat1
  use rrsw_kg17, only: q17_rayl => rayl
  use rrsw_kg17, only: q17_strrat => strrat
  use rrsw_kg18, only: q18_rayl => rayl
  use rrsw_kg18, only: q18_strrat => strrat
  use rrsw_kg19, only: q19_rayl => rayl
  use rrsw_kg19, only: q19_strrat => strrat
  use rrsw_kg20, only: q20_rayl => rayl
  use rrsw_kg21, only: q21_rayl => rayl
  use rrsw_kg21, only: q21_strrat => strrat
  use rrsw_kg22, only: q22_rayl => rayl
  use rrsw_kg22, only: q22_strrat => strrat
  use rrsw_kg23, only: q23_givfac => givfac
  use rrsw_kg24, only: q24_strrat => strrat
  use rrsw_kg27, only: q27_scalekur => scalekur
  use rrsw_kg28, only: q28_rayl => rayl
  use rrsw_kg28, only: q28_strrat => strrat
  use rrsw_kg29, only: q29_rayl => rayl
  use rrsw_kg16, only: q16_layreffr => layreffr
  use rrsw_kg17, only: q17_layreffr => layreffr
  use rrsw_kg18, only: q18_layreffr => layreffr
  use rrsw_kg19, only: q19_layreffr => layreffr
  use rrsw_kg20, only: q20_layreffr => layreffr
  use rrsw_kg21, only: q21_layreffr => layreffr
  use rrsw_kg22, only: q22_layreffr => layreffr
  use rrsw_kg23, only: q23_layreffr => layreffr
  use rrsw_kg24, only: q24_layreffr => layreffr
  use rrsw_kg25, only: q25_layreffr => layreffr
  use rrsw_kg27, only: q27_layreffr => layreffr
  use rrsw_kg28, only: q28_layreffr => layreffr
  use rrsw_kg29, only: q29_layreffr => layreffr
  implicit none
  character(kind=c_char), intent(in) :: cname(*)
  real(c_double) :: v
  character(len=64) :: name
  integer :: i
  name = ' '
  do i = 1, 64
    if (cname(i) == c_null_char) exit
    name(i:i) = cname(i)
  enddo
  v = -999.0
  select case (trim(name))
  case ('kg16.rayl'); v = q16_rayl
  case ('kg16.strrat1'); v = q16_strrat1
  case ('kg17.rayl'); v = q17_rayl
  case ('kg17.strrat'); v = q17_strrat
  case ('kg18.rayl'); v = q18_rayl
  case ('kg18.strrat'); v = q18_strrat
  case ('kg19.rayl'); v = q19_rayl
  case ('kg19.strrat'); v = q19_strrat
  case ('kg20.rayl'); v = q20_rayl
  case ('kg21.rayl'); v = q21_rayl
  case ('kg21.strrat'); v = q21_strrat
  case ('kg22.rayl'); v = q22_rayl
  case ('kg22.strrat'); v = q22_strrat
  case ('kg23.givfac'); v = q23_givfac
  case ('kg24.strrat'); v = q24_strrat
  case ('kg27.scalekur'); v = q27_scalekur
  case ('kg28.rayl'); v = q28_rayl
  case ('kg28.strrat'); v = q28_strrat
  case ('kg29.rayl'); v = q29_rayl
  case ('kg16.layreffr'); v = q16_layreffr
  case ('kg17.layreffr'); v = q17_layreffr
  case ('kg18.layreffr'); v = q18_layreffr
  case ('kg19.layreffr'); v = q19_layreffr
  case ('kg20.layreffr'); v = q20_layreffr
  case ('kg21.layreffr'); v = q21_layreffr
  case ('kg22.layreffr'); v = q22_layreffr
  case ('kg23.layreffr'); v = q23_layreffr
  case ('kg24.layreffr'); v = q24_layreffr
  case ('kg25.layreffr'); v = q25_layreffr
  case ('kg27.layreffr'); v = q27_layreffr
  case ('kg28.layreffr'); v = q28_layreffr
  case ('kg29.layreffr'); v = q29_layreffr
  end select
end function cm1f_rrtmg_sw_scalar

function cm1f_rrtmg_sw_table(cname, out, n) result(cnt) bind(C, name="cm1f_rrtmg_sw_table")
  use iso_c_binding
  use rrsw_aer, only: t_rsrtaua => rsrtaua
  use rrsw_aer, only: t_rsrpiza => rsrpiza
  use rrsw_aer, only: t_rsrasya => rsrasya
  use rrsw_cld, only: t_extliq1 => extliq1
  use rrsw_cld, only: t_ssaliq1 => ssaliq1
  use rrsw_cld, only: t_asyliq1 => asyliq1
  use rrsw_cld, only: t_extice2 => extice2
  use rrsw_cld, only: t_ssaice2 => ssaice2
  use rrsw_cld, only: t_asyice2 => asyice2
  use rrsw_cld, only: t_extice3 => extice3
  use rrsw_cld, only: t_ssaice3 => ssaice3
  use rrsw_cld, only: t_asyice3 => asyice3
  use rrsw_cld, only: t_fdlice3 => fdlice3
  use rrsw_cld, only: t_abari => abari
  use rrsw_cld, only: t_bbari => bbari
  use rrsw_cld, only: t_cbari => cbari
  use rrsw_cld, only: t_dbari => dbari
  use rrsw_cld, only: t_ebari => ebari
  use rrsw_cld, only: t_fbari => fbari
  use rrsw_ref, only: t_pref => pref
  use rrsw_ref, only: t_preflog => preflog
  use rrsw_ref, only: t_tref => tref
  use rrsw_tbl, only: t_exp_tbl => exp_tbl
  use rrsw_wvn, only: t_ng => ng
  use rrsw_wvn, only: t_nspa => nspa
  use rrsw_wvn, only: t_nspb => nspb
  use rrsw_wvn, only: t_wavenum1 => wavenum1
  use rrsw_wvn, only: t_wavenum2 => wavenum2
  use rrsw_wvn, only: t_delwave => delwave
  use rrsw_wvn, only: t_ngc => ngc
  use rrsw_wvn, only: t_ngs => ngs
  use rrsw_wvn, only: t_ngn => ngn
  use rrsw_wvn, only: t_ngb => ngb
  use rrsw_wvn, only: t_ngm => ngm
  use rrsw_wvn, only: t_wt => wt
  use rrsw_wvn, only: t_rwgt => rwgt
  use rrsw_kg16, only: skg16_absa => absa
  use rrsw_kg16, only: skg16_absb => absb
  use rrsw_kg16, only: skg16_selfref => selfref
  use rrsw_kg16, only: skg16_forref => forref
  use rrsw_kg16, only: skg16_sfluxref => sfluxref
  use rrsw_kg17, only: skg17_absa => absa
  use rrsw_kg17, only: skg17_absb => absb
  use rrsw_kg17, only: skg17_selfref => selfref
  use rrsw_kg17, only: skg17_forref => forref
  use rrsw_kg17, only: skg17_sfluxref => sfluxref
  use rrsw_kg18, only: skg18_absa => absa
  use rrsw_kg18, only: skg18_absb => absb
  use rrsw_kg18, only: skg18_selfref => selfref
  use rrsw_kg18, only: skg18_forref => forref
  use rrsw_kg18, only: skg18_sfluxref => sfluxref
  use rrsw_kg19, only: skg19_absa => absa
  use rrsw_kg19, only: skg19_absb => absb
  use rrsw_kg19, only: skg19_selfref => selfref
  use rrsw_kg19, only: skg19_forref => forref
  use rrsw_kg19, only: skg19_sfluxref => sfluxref
  use rrsw_kg20, only: skg20_absch4 => absch4
  use rrsw_kg20, only: skg20_absa => absa
  use rrsw_kg20, only: skg20_absb => absb
  use rrsw_kg20, only: skg20_selfref => selfref
  use rrsw_kg20, only: skg20_forref => forref
  use rrsw_kg20, only: skg20_sfluxref => sfluxref
  use rrsw_kg21, only: skg21_absa => absa
  use rrsw_kg21, only: skg21_absb => absb
  use rrsw_kg21, only: skg21_selfref => selfref
  use rrsw_kg21, only: skg21_forref => forref
  use rrsw_kg21, only: skg21_sfluxref => sfluxref
  use rrsw_kg22, only: skg22_absa => absa
  use rrsw_kg22, only: skg22_absb => absb
  use rrsw_kg22, only: skg22_selfref => selfref
  use rrsw_kg22, only: skg22_forref => forref
  use rrsw_kg22, only: skg22_sfluxref => sfluxref
  use rrsw_kg23, only: skg23_rayl => rayl
  use rrsw_kg23, only: skg23_absa => absa
  use rrsw_kg23, only: skg23_selfref => selfref
  use rrsw_kg23, only: skg23_forref => forref
  use rrsw_kg23, only: skg23_sfluxref => sfluxref
  use rrsw_kg24, only: skg24_rayla => rayla
  use rrsw_kg24, only: skg24_raylb => raylb
  use rrsw_kg24, only: skg24_abso3a => abso3a
  use rrsw_kg24, only: skg24_abso3b => abso3b
  use rrsw_kg24, only: skg24_absa => absa
  use rrsw_kg24, only: skg24_absb => absb
  use rrsw_kg24, only: skg24_selfref => selfref
  use rrsw_kg24, only: skg24_forref => forref
  use rrsw_kg24, only: skg24_sfluxref => sfluxref
  use rrsw_kg25, only: skg25_rayl => rayl
  use rrsw_kg25, only: skg25_abso3a => abso3a
  use rrsw_kg25, only: skg25_abso3b => abso3b
  use rrsw_kg25, only: skg25_absa => absa
  use rrsw_kg25, only: skg25_sfluxref => sfluxref
  use rrsw_kg26, only: skg26_rayl => rayl
  use rrsw_kg26, only: skg26_sfluxref => sfluxref
  use rrsw_kg27, only: skg27_rayl => rayl
  use rrsw_kg27, only: skg27_absa => absa
  use rrsw_kg27, only: skg27_absb => absb
  use rrsw_kg27, only: skg27_sfluxref => sfluxref
  use rrsw_kg28, only: skg28_absa => absa
  use rrsw_kg28, only: skg28_absb => absb
  use rrsw_kg28, only: skg28_sfluxref => sfluxref
  use rrsw_kg29, only: skg29_absh2o => absh2o
  use rrsw_kg29, only: skg29_absco2 => absco2
  use rrsw_kg29, only: skg29_absa => absa
  use rrsw_kg29, only: skg29_absb => absb
  use rrsw_kg29, only: skg29_selfref => selfref
  use rrsw_kg29, only: skg29_forref => forref
  use rrsw_kg29, only: skg29_sfluxref => sfluxref
  use module_ra_rrtmg_lw, only: rad_retab => retab
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
  case ('rsrtaua'); call put_r(t_rsrtaua)
  case ('rsrpiza'); call put_r(t_rsrpiza)
  case ('rsrasya'); call put_r(t_rsrasya)
  case ('extliq1'); call put_r(t_extliq1)
  case ('ssaliq1'); call put_r(t_ssaliq1)
  case ('asyliq1'); call put_r(t_asyliq1)
  case ('extice2'); call put_r(t_extice2)
  case ('ssaice2'); call put_r(t_ssaice2)
  case ('asyice2'); call put_r(t_asyice2)
  case ('extice3'); call put_r(t_extice3)
  case ('ssaice3'); call put_r(t_ssaice3)
  case ('asyice3'); call put_r(t_asyice3)
  case ('fdlice3'); call put_r(t_fdlice3)
  case ('abari'); call put_r(t_abari)
  case ('bbari'); call put_r(t_bbari)
  case ('cbari'); call put_r(t_cbari)
  case ('dbari'); call put_r(t_dbari)
  case ('ebari'); call put_r(t_ebari)
  case ('fbari'); call put_r(t_fbari)
  case ('pref'); call put_r(t_pref)
  case ('preflog'); call put_r(t_preflog)
  case ('tref'); call put_r(t_tref)
  case ('exp_tbl'); call put_r(t_exp_tbl)
  case ('wavenum1'); call put_r(t_wavenum1)
  case ('wavenum2'); call put_r(t_wavenum2)
  case ('delwave'); call put_r(t_delwave)
  case ('wt'); call put_r(t_wt)
  case ('rwgt'); call put_r(t_rwgt)
  case ('rad.retab'); call put_r(rad_retab)
  case ('kg16.absa'); call put_r(skg16_absa)
  case ('kg16.absb'); call put_r(skg16_absb)
  case ('kg16.selfref'); call put_r(skg16_selfref)
  case ('kg16.forref'); call put_r(skg16_forref)
  case ('kg16.sfluxref'); call put_r(skg16_sfluxref)
  case ('kg17.absa'); call put_r(skg17_absa)
  case ('kg17.absb'); call put_r(skg17_absb)
  case ('kg17.selfref'); call put_r(skg17_selfref)
  case ('kg17.forref'); call put_r(skg17_forref)
  case ('kg17.sfluxref'); call put_r(skg17_sfluxref)
  case ('kg18.absa'); call put_r(skg18_absa)
  case ('kg18.absb'); call put_r(skg18_absb)
  case ('kg18.selfref'); call put_r(skg18_selfref)
  case ('kg18.forref'); call put_r(skg18_forref)
  case ('kg18.sfluxref'); call put_r(skg18_sfluxref)
  case ('kg19.absa'); call put_r(skg19_absa)
  case ('kg19.absb'); call put_r(skg19_absb)
  case ('kg19.selfref'); call put_r(skg19_selfref)
  case ('kg19.forref'); call put_r(skg19_forref)
  case ('kg19.sfluxref'); call put_r(skg19_sfluxref)
  case ('kg20.absch4'); call put_r(skg20_absch4)
  case ('kg20.absa'); call put_r(skg20_absa)
  case ('kg20.absb'); call put_r(skg20_absb)
  case ('kg20.selfref'); call put_r(skg20_selfref)
  case ('kg20.forref'); call put_r(skg20_forref)
  case ('kg20.sfluxref'); call put_r(skg20_sfluxref)
  case ('kg21.absa'); call put_r(skg21_absa)
  case ('kg21.absb'); call put_r(skg21_absb)
  case ('kg21.selfref'); call put_r(skg21_selfref)
  case ('kg21.forref'); call put_r(skg21_forref)
  case ('kg21.sfluxref'); call put_r(skg21_sfluxref)
  case ('kg22.absa'); call put_r(skg22_absa)
  case ('kg22.absb'); call put_r(skg22_absb)
  case ('kg22.selfref'); call put_r(skg22_selfref)
  case ('kg22.forref'); call put_r(skg22_forref)
  case ('kg22.sfluxref'); call put_r(skg22_sfluxref)
  case ('kg23.rayl'); call put_r(skg23_rayl)
  case ('kg23.absa'); call put_r(skg23_absa)
  case ('kg23.selfref'); call put_r(skg23_selfref)
  case ('kg23.forref'); call put_r(skg23_forref)
  case ('kg23.sfluxref'); call put_r(skg23_sfluxref)
  case ('kg24.rayla'); call put_r(skg24_rayla)
  case ('kg24.raylb'); call put_r(skg24_raylb)
  case ('kg24.abso3a'); call put_r(skg24_abso3a)
  case ('kg24.abso3b'); call put_r(skg24_abso3b)
  case ('kg24.absa'); call put_r(skg24_absa)
  case ('kg24.absb'); call put_r(skg24_absb)
  case ('kg24.selfref'); call put_r(skg24_selfref)
  case ('kg24.forref'); call put_r(skg24_forref)
  case ('kg24.sfluxref'); call put_r(skg24_sfluxref)
  case ('kg25.rayl'); call put_r(skg25_rayl)
  case ('kg25.abso3a'); call put_r(skg25_abso3a)
  case ('kg25.abso3b'); call put_r(skg25_abso3b)
  case ('kg25.absa'); call put_r(skg25_absa)
  case ('kg25.sfluxref'); call put_r(skg25_sfluxref)
  case ('kg26.rayl'); call put_r(skg26_rayl)
  case ('kg26.sfluxref'); call put_r(skg26_sfluxref)
  case ('kg27.rayl'); call put_r(skg27_rayl)
  case ('kg27.absa'); call put_r(skg27_absa)
  case ('kg27.absb'); call put_r(skg27_absb)
  case ('kg27.sfluxref'); call put_r(skg27_sfluxref)
  case ('kg28.absa'); call put_r(skg28_absa)
  case ('kg28.absb'); call put_r(skg28_absb)
  case ('kg28.sfluxref'); call put_r(skg28_sfluxref)
  case ('kg29.absh2o'); call put_r(skg29_absh2o)
  case ('kg29.absco2'); call put_r(skg29_absco2)
  case ('kg29.absa'); call put_r(skg29_absa)
  case ('kg29.absb'); call put_r(skg29_absb)
  case ('kg29.selfref'); call put_r(skg29_selfref)
  case ('kg29.forref'); call put_r(skg29_forref)
  case ('kg29.sfluxref'); call put_r(skg29_sfluxref)
  case ('ng'); call put_i(t_ng)
  case ('nspa'); call put_i(t_nspa)
  case ('nspb'); call put_i(t_nspb)
  case ('ngc'); call put_i(t_ngc)
  case ('ngs'); call put_i(t_ngs)
  case ('ngn'); call put_i(t_ngn)
  case ('ngb'); call put_i(t_ngb)
  case ('ngm'); call put_i(t_ngm)
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
end function cm1f_rrtmg_sw_table
