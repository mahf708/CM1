! Harness driver for src/module_ra_rrtmg_lw.F (RRTMG longwave as CM1 uses it).
! Needs RRTMG_LW_DATA in the working directory.
! params = [mode, ncol, nlay, p_top, yr, permuteseed]
!   mode 1: rrtmg_lwinit, dump the reduced tables (b<NN>_<name>)
!   mode 2: mcica_subcol_lw + the rrtmg_lw chain (inatm, cldprmc, setcoef,
!           taumol, rtrnmc) per column, dumping every intermediate, and
!           rrtmg_lw itself on all columns
!   mode 3: RRTMG_LWRAD on ncol columns (nk = nlay model layers)
program drv_rrtmg_lw
  use harness_io
  use parkind, only : im => kind_im, rb => kind_rb
  use parrrtm, only : nbndlw, ngptlw, mxmol, maxxsec
  use rrlw_tbl, only : tau_tbl, exp_tbl, tfn_tbl
  use rrlw_wvn, only : totplnk, rwgt
  use rrlw_cld, only : absliq1, absice3
  use rrlw_ref, only : chi_mls
  use rrtmg_lw_rad, only : rrtmg_lw, inatm
  use rrtmg_lw_cldprmc, only : cldprmc
  use rrtmg_lw_setcoef, only : setcoef
  use rrtmg_lw_taumol, only : taumol
  use rrtmg_lw_rtrnmc, only : rtrnmc
  use mcica_subcol_gen_lw, only : mcica_subcol_lw
  use module_ra_rrtmg_lw, only : rrtmg_lwinit, rrtmg_lwrad
  implicit none
  real :: params(6), p_top
  integer :: mode, ncol, nlay, yr, ic, k, permuteseed, irng, icldv
  ! mode 2 inputs [ncol, nlay(+1)]
  real, allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:), h2ovmr(:,:), &
       o3vmr(:,:), co2vmr(:,:), ch4vmr(:,:), n2ovmr(:,:), o2vmr(:,:), cfc11vmr(:,:), &
       cfc12vmr(:,:), cfc22vmr(:,:), ccl4vmr(:,:), emis(:,:), cldfrac(:,:), ciwp(:,:), &
       clwp(:,:), cswp(:,:), rei(:,:), rel(:,:), res(:,:), hgt(:,:), taucld(:,:,:), tauaer(:,:,:)
  real, allocatable :: cldfmcl(:,:,:), ciwpmcl(:,:,:), clwpmcl(:,:,:), cswpmcl(:,:,:), &
       taucmcl(:,:,:), reicmcl(:,:), relqmcl(:,:), resnmcl(:,:)
  real, allocatable :: uflx(:,:), dflx(:,:), hr(:,:), uflxc(:,:), dflxc(:,:), hrc(:,:), &
       uflxcln(:,:), dflxcln(:,:)
  ! per-column chain
  integer :: nlayers, inflag, iceflag, liqflag, ncbands, laytrop
  real :: tbound, pwvcm
  real, allocatable :: pavel(:), tavel(:), pz(:), tz(:), coldry(:), wbrodl(:), wkl(:,:), wx(:,:), &
       semiss(:), cldfmc(:,:), taucmc(:,:), ciwpmc(:,:), clwpmc(:,:), cswpmc(:,:), reicmc(:), &
       relqmc(:), resnmc(:), taua(:,:), planklay(:,:), planklev(:,:), plankbnd(:), &
       colh2o(:), colco2(:), colo3(:), coln2o(:), colco(:), colch4(:), colo2(:), colbrd(:), &
       fac00(:), fac01(:), fac10(:), fac11(:), rat_h2oco2(:), rat_h2oco2_1(:), rat_h2oo3(:), &
       rat_h2oo3_1(:), rat_h2on2o(:), rat_h2on2o_1(:), rat_h2och4(:), rat_h2och4_1(:), &
       rat_n2oco2(:), rat_n2oco2_1(:), rat_o3co2(:), rat_o3co2_1(:), selffac(:), selffrac(:), &
       forfac(:), forfrac(:), minorfrac(:), scaleminor(:), scaleminorn2(:), fracs(:,:), taug(:,:), &
       totuflux(:), totdflux(:), fnet(:), htr(:), totuclfl(:), totdclfl(:), fnetc(:), htrc(:)
  integer, allocatable :: jp(:), jt(:), jt1(:), indself(:), indfor(:), indminor(:)
  ! dumps
  real, allocatable :: o_coldry(:,:), o_wbrodl(:,:), o_wkl(:,:,:), o_wx(:,:,:), o_pwvcm(:), &
       o_taucmc(:,:,:), o_lay(:,:,:), o_ilay(:,:,:), o_laytrop(:), o_planklay(:,:,:), &
       o_planklev(:,:,:), o_plankbnd(:,:), o_fracs(:,:,:), o_taug(:,:,:), o_flx(:,:,:)
  ! mode 3
  real, allocatable :: p8w(:,:,:), p3d(:,:,:), pi3d(:,:,:), dz8w(:,:,:), t3d(:,:,:), t8w(:,:,:), &
       rho3d(:,:,:), cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), qi3d(:,:,:), &
       qs3d(:,:,:), qg3d(:,:,:), o33d(:,:,:), rthratenlw(:,:,:), lwtenc(:,:,:), &
       re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
  real, allocatable :: tsk(:,:), emiss(:,:), xland(:,:), xice(:,:), snow(:,:), glw(:,:), olr(:,:), &
       lwcf(:,:), lwupt(:,:), lwuptc(:,:), lwdnt(:,:), lwdntc(:,:), lwupb(:,:), lwupbc(:,:), &
       lwdnb(:,:), lwdnbc(:,:), lwuptcln(:,:), lwdntcln(:,:), lwupbcln(:,:), lwdnbcln(:,:)
  real :: julian, co2out, b1(1)

  call rd('params', params, size(params))
  mode = nint(params(1)); ncol = nint(params(2)); nlay = nint(params(3))
  p_top = params(4); yr = nint(params(5)); permuteseed = nint(params(6))
  if (mode == 3) then
    call rrtmg_lwinit(p_top, .true., 0, 1, ncol+1, 1, 2, 1, nlay+1, &
         1, ncol, 1, 1, 1, nlay+1, 1, ncol, 1, 1, 1, nlay)
  else
    call rrtmg_lwinit(p_top, .true., 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1)
  end if

  select case (mode)
  case (1)
    call dump01
    call dump02
    call dump03
    call dump04
    call dump05
    call dump06
    call dump07
    call dump08
    call dump09
    call dump10
    call dump11
    call dump12
    call dump13
    call dump14
    call dump15
    call dump16
    call wr('tau_tbl', tau_tbl, size(tau_tbl)); call wr('exp_tbl', exp_tbl, size(exp_tbl))
    call wr('tfn_tbl', tfn_tbl, size(tfn_tbl)); call wr('totplnk', totplnk, size(totplnk))
    call wr('absliq1', absliq1, size(absliq1)); call wr('absice3', absice3, size(absice3))
    call wr('chi_mls', chi_mls, size(chi_mls)); call wr('rwgt', rwgt, size(rwgt))
  case (2)
    call chain
  case (3)
    call lwrad
  end select

contains
  subroutine dump01
    use rrlw_kg01, only : fracrefa, fracrefb, ka, kb, ka_mn2, kb_mn2, selfref, forref, absa, absb
    call wr('b01_fracrefa', fracrefa, size(fracrefa))
    call wr('b01_fracrefb', fracrefb, size(fracrefb))
    call wr('b01_ka', ka, size(ka))
    call wr('b01_kb', kb, size(kb))
    call wr('b01_ka_mn2', ka_mn2, size(ka_mn2))
    call wr('b01_kb_mn2', kb_mn2, size(kb_mn2))
    call wr('b01_selfref', selfref, size(selfref))
    call wr('b01_forref', forref, size(forref))
    call wr('b01_absa', absa, size(absa))
    call wr('b01_absb', absb, size(absb))
  end subroutine dump01
  subroutine dump02
    use rrlw_kg02, only : fracrefa, fracrefb, ka, kb, selfref, forref, absa, absb
    call wr('b02_fracrefa', fracrefa, size(fracrefa))
    call wr('b02_fracrefb', fracrefb, size(fracrefb))
    call wr('b02_ka', ka, size(ka))
    call wr('b02_kb', kb, size(kb))
    call wr('b02_selfref', selfref, size(selfref))
    call wr('b02_forref', forref, size(forref))
    call wr('b02_absa', absa, size(absa))
    call wr('b02_absb', absb, size(absb))
  end subroutine dump02
  subroutine dump03
    use rrlw_kg03, only : fracrefa, fracrefb, ka, kb, ka_mn2o, kb_mn2o, selfref, forref, absa, absb
    call wr('b03_fracrefa', fracrefa, size(fracrefa))
    call wr('b03_fracrefb', fracrefb, size(fracrefb))
    call wr('b03_ka', ka, size(ka))
    call wr('b03_kb', kb, size(kb))
    call wr('b03_ka_mn2o', ka_mn2o, size(ka_mn2o))
    call wr('b03_kb_mn2o', kb_mn2o, size(kb_mn2o))
    call wr('b03_selfref', selfref, size(selfref))
    call wr('b03_forref', forref, size(forref))
    call wr('b03_absa', absa, size(absa))
    call wr('b03_absb', absb, size(absb))
  end subroutine dump03
  subroutine dump04
    use rrlw_kg04, only : fracrefa, fracrefb, ka, kb, selfref, forref, absa, absb
    call wr('b04_fracrefa', fracrefa, size(fracrefa))
    call wr('b04_fracrefb', fracrefb, size(fracrefb))
    call wr('b04_ka', ka, size(ka))
    call wr('b04_kb', kb, size(kb))
    call wr('b04_selfref', selfref, size(selfref))
    call wr('b04_forref', forref, size(forref))
    call wr('b04_absa', absa, size(absa))
    call wr('b04_absb', absb, size(absb))
  end subroutine dump04
  subroutine dump05
    use rrlw_kg05, only : fracrefa, fracrefb, ka, kb, ka_mo3, ccl4, selfref, forref, absa, absb
    call wr('b05_fracrefa', fracrefa, size(fracrefa))
    call wr('b05_fracrefb', fracrefb, size(fracrefb))
    call wr('b05_ka', ka, size(ka))
    call wr('b05_kb', kb, size(kb))
    call wr('b05_ka_mo3', ka_mo3, size(ka_mo3))
    call wr('b05_ccl4', ccl4, size(ccl4))
    call wr('b05_selfref', selfref, size(selfref))
    call wr('b05_forref', forref, size(forref))
    call wr('b05_absa', absa, size(absa))
    call wr('b05_absb', absb, size(absb))
  end subroutine dump05
  subroutine dump06
    use rrlw_kg06, only : fracrefa, ka, ka_mco2, cfc11adj, cfc12, selfref, forref, absa
    call wr('b06_fracrefa', fracrefa, size(fracrefa))
    call wr('b06_ka', ka, size(ka))
    call wr('b06_ka_mco2', ka_mco2, size(ka_mco2))
    call wr('b06_cfc11adj', cfc11adj, size(cfc11adj))
    call wr('b06_cfc12', cfc12, size(cfc12))
    call wr('b06_selfref', selfref, size(selfref))
    call wr('b06_forref', forref, size(forref))
    call wr('b06_absa', absa, size(absa))
  end subroutine dump06
  subroutine dump07
    use rrlw_kg07, only : fracrefa, fracrefb, ka, kb, ka_mco2, kb_mco2, selfref, forref, absa, absb
    call wr('b07_fracrefa', fracrefa, size(fracrefa))
    call wr('b07_fracrefb', fracrefb, size(fracrefb))
    call wr('b07_ka', ka, size(ka))
    call wr('b07_kb', kb, size(kb))
    call wr('b07_ka_mco2', ka_mco2, size(ka_mco2))
    call wr('b07_kb_mco2', kb_mco2, size(kb_mco2))
    call wr('b07_selfref', selfref, size(selfref))
    call wr('b07_forref', forref, size(forref))
    call wr('b07_absa', absa, size(absa))
    call wr('b07_absb', absb, size(absb))
  end subroutine dump07
  subroutine dump08
    use rrlw_kg08, only : fracrefa, fracrefb, ka, kb, ka_mco2, kb_mco2, ka_mn2o, kb_mn2o, ka_mo3, cfc12, cfc22adj, selfref, forref, absa, absb
    call wr('b08_fracrefa', fracrefa, size(fracrefa))
    call wr('b08_fracrefb', fracrefb, size(fracrefb))
    call wr('b08_ka', ka, size(ka))
    call wr('b08_kb', kb, size(kb))
    call wr('b08_ka_mco2', ka_mco2, size(ka_mco2))
    call wr('b08_kb_mco2', kb_mco2, size(kb_mco2))
    call wr('b08_ka_mn2o', ka_mn2o, size(ka_mn2o))
    call wr('b08_kb_mn2o', kb_mn2o, size(kb_mn2o))
    call wr('b08_ka_mo3', ka_mo3, size(ka_mo3))
    call wr('b08_cfc12', cfc12, size(cfc12))
    call wr('b08_cfc22adj', cfc22adj, size(cfc22adj))
    call wr('b08_selfref', selfref, size(selfref))
    call wr('b08_forref', forref, size(forref))
    call wr('b08_absa', absa, size(absa))
    call wr('b08_absb', absb, size(absb))
  end subroutine dump08
  subroutine dump09
    use rrlw_kg09, only : fracrefa, fracrefb, ka, kb, ka_mn2o, kb_mn2o, selfref, forref, absa, absb
    call wr('b09_fracrefa', fracrefa, size(fracrefa))
    call wr('b09_fracrefb', fracrefb, size(fracrefb))
    call wr('b09_ka', ka, size(ka))
    call wr('b09_kb', kb, size(kb))
    call wr('b09_ka_mn2o', ka_mn2o, size(ka_mn2o))
    call wr('b09_kb_mn2o', kb_mn2o, size(kb_mn2o))
    call wr('b09_selfref', selfref, size(selfref))
    call wr('b09_forref', forref, size(forref))
    call wr('b09_absa', absa, size(absa))
    call wr('b09_absb', absb, size(absb))
  end subroutine dump09
  subroutine dump10
    use rrlw_kg10, only : fracrefa, fracrefb, ka, kb, selfref, forref, absa, absb
    call wr('b10_fracrefa', fracrefa, size(fracrefa))
    call wr('b10_fracrefb', fracrefb, size(fracrefb))
    call wr('b10_ka', ka, size(ka))
    call wr('b10_kb', kb, size(kb))
    call wr('b10_selfref', selfref, size(selfref))
    call wr('b10_forref', forref, size(forref))
    call wr('b10_absa', absa, size(absa))
    call wr('b10_absb', absb, size(absb))
  end subroutine dump10
  subroutine dump11
    use rrlw_kg11, only : fracrefa, fracrefb, ka, kb, ka_mo2, kb_mo2, selfref, forref, absa, absb
    call wr('b11_fracrefa', fracrefa, size(fracrefa))
    call wr('b11_fracrefb', fracrefb, size(fracrefb))
    call wr('b11_ka', ka, size(ka))
    call wr('b11_kb', kb, size(kb))
    call wr('b11_ka_mo2', ka_mo2, size(ka_mo2))
    call wr('b11_kb_mo2', kb_mo2, size(kb_mo2))
    call wr('b11_selfref', selfref, size(selfref))
    call wr('b11_forref', forref, size(forref))
    call wr('b11_absa', absa, size(absa))
    call wr('b11_absb', absb, size(absb))
  end subroutine dump11
  subroutine dump12
    use rrlw_kg12, only : fracrefa, ka, selfref, forref, absa
    call wr('b12_fracrefa', fracrefa, size(fracrefa))
    call wr('b12_ka', ka, size(ka))
    call wr('b12_selfref', selfref, size(selfref))
    call wr('b12_forref', forref, size(forref))
    call wr('b12_absa', absa, size(absa))
  end subroutine dump12
  subroutine dump13
    use rrlw_kg13, only : fracrefa, fracrefb, ka, ka_mco2, ka_mco, kb_mo3, selfref, forref, absa
    call wr('b13_fracrefa', fracrefa, size(fracrefa))
    call wr('b13_fracrefb', fracrefb, size(fracrefb))
    call wr('b13_ka', ka, size(ka))
    call wr('b13_ka_mco2', ka_mco2, size(ka_mco2))
    call wr('b13_ka_mco', ka_mco, size(ka_mco))
    call wr('b13_kb_mo3', kb_mo3, size(kb_mo3))
    call wr('b13_selfref', selfref, size(selfref))
    call wr('b13_forref', forref, size(forref))
    call wr('b13_absa', absa, size(absa))
  end subroutine dump13
  subroutine dump14
    use rrlw_kg14, only : fracrefa, fracrefb, ka, kb, selfref, forref, absa, absb
    call wr('b14_fracrefa', fracrefa, size(fracrefa))
    call wr('b14_fracrefb', fracrefb, size(fracrefb))
    call wr('b14_ka', ka, size(ka))
    call wr('b14_kb', kb, size(kb))
    call wr('b14_selfref', selfref, size(selfref))
    call wr('b14_forref', forref, size(forref))
    call wr('b14_absa', absa, size(absa))
    call wr('b14_absb', absb, size(absb))
  end subroutine dump14
  subroutine dump15
    use rrlw_kg15, only : fracrefa, ka, ka_mn2, selfref, forref, absa
    call wr('b15_fracrefa', fracrefa, size(fracrefa))
    call wr('b15_ka', ka, size(ka))
    call wr('b15_ka_mn2', ka_mn2, size(ka_mn2))
    call wr('b15_selfref', selfref, size(selfref))
    call wr('b15_forref', forref, size(forref))
    call wr('b15_absa', absa, size(absa))
  end subroutine dump15
  subroutine dump16
    use rrlw_kg16, only : fracrefa, fracrefb, ka, kb, selfref, forref, absa, absb
    call wr('b16_fracrefa', fracrefa, size(fracrefa))
    call wr('b16_fracrefb', fracrefb, size(fracrefb))
    call wr('b16_ka', ka, size(ka))
    call wr('b16_kb', kb, size(kb))
    call wr('b16_selfref', selfref, size(selfref))
    call wr('b16_forref', forref, size(forref))
    call wr('b16_absa', absa, size(absa))
    call wr('b16_absb', absb, size(absb))
  end subroutine dump16

  subroutine chain
    integer :: nl
    nl = nlay
    allocate(play(ncol,nl), plev(ncol,nl+1), tlay(ncol,nl), tlev(ncol,nl+1), tsfc(ncol), &
         h2ovmr(ncol,nl), o3vmr(ncol,nl), co2vmr(ncol,nl), ch4vmr(ncol,nl), n2ovmr(ncol,nl), &
         o2vmr(ncol,nl), cfc11vmr(ncol,nl), cfc12vmr(ncol,nl), cfc22vmr(ncol,nl), ccl4vmr(ncol,nl), &
         emis(ncol,nbndlw), cldfrac(ncol,nl), ciwp(ncol,nl), clwp(ncol,nl), cswp(ncol,nl), &
         rei(ncol,nl), rel(ncol,nl), res(ncol,nl), hgt(ncol,nl), taucld(nbndlw,ncol,nl), &
         tauaer(ncol,nl,nbndlw))
    allocate(cldfmcl(ngptlw,ncol,nl), ciwpmcl(ngptlw,ncol,nl), clwpmcl(ngptlw,ncol,nl), &
         cswpmcl(ngptlw,ncol,nl), taucmcl(ngptlw,ncol,nl), reicmcl(ncol,nl), relqmcl(ncol,nl), &
         resnmcl(ncol,nl))
    allocate(uflx(ncol,nl+1), dflx(ncol,nl+1), hr(ncol,nl), uflxc(ncol,nl+1), dflxc(ncol,nl+1), &
         hrc(ncol,nl), uflxcln(ncol,nl+1), dflxcln(ncol,nl+1))
    call rd('play', play, size(play)); call rd('plev', plev, size(plev))
    call rd('tlay', tlay, size(tlay)); call rd('tlev', tlev, size(tlev))
    call rd('tsfc', tsfc, size(tsfc)); call rd('h2ovmr', h2ovmr, size(h2ovmr))
    call rd('o3vmr', o3vmr, size(o3vmr)); call rd('co2vmr', co2vmr, size(co2vmr))
    call rd('ch4vmr', ch4vmr, size(ch4vmr)); call rd('n2ovmr', n2ovmr, size(n2ovmr))
    call rd('o2vmr', o2vmr, size(o2vmr)); call rd('cfc11vmr', cfc11vmr, size(cfc11vmr))
    call rd('cfc12vmr', cfc12vmr, size(cfc12vmr)); call rd('cfc22vmr', cfc22vmr, size(cfc22vmr))
    call rd('ccl4vmr', ccl4vmr, size(ccl4vmr)); call rd('emis', emis, size(emis))
    call rd('cldfrac', cldfrac, size(cldfrac)); call rd('ciwp', ciwp, size(ciwp))
    call rd('clwp', clwp, size(clwp)); call rd('rei', rei, size(rei)); call rd('rel', rel, size(rel))
    cswp = 0.0; res = 10.0; hgt = 0.0; taucld = 0.0; tauaer = 0.0
    irng = 0
    call mcica_subcol_lw(1, ncol, nl, 2, permuteseed, irng, play, hgt, &
         cldfrac, ciwp, clwp, cswp, rei, rel, res, taucld, cldfmcl, &
         ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, taucmcl)
    call wr('cldfmcl', cldfmcl, size(cldfmcl)); call wr('ciwpmcl', ciwpmcl, size(ciwpmcl))
    call wr('clwpmcl', clwpmcl, size(clwpmcl)); call wr('taucmcl', taucmcl, size(taucmcl))
    ! rrtmg_lw itself
    icldv = 2
    call rrtmg_lw(ncol, nl, icldv, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, &
         n2ovmr, o2vmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, 2, 3, 1, cldfmcl, &
         taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, tauaer, &
         uflx, dflx, hr, uflxc, dflxc, hrc, uflxcln, dflxcln, 0)
    call wr('uflx', uflx, size(uflx)); call wr('dflx', dflx, size(dflx)); call wr('hr', hr, size(hr))
    call wr('uflxc', uflxc, size(uflxc)); call wr('dflxc', dflxc, size(dflxc)); call wr('hrc', hrc, size(hrc))
    ! the chain, one column at a time, with every intermediate
    allocate(pavel(nl+1), tavel(nl+1), pz(0:nl+1), tz(0:nl+1), coldry(nl+1), wbrodl(nl+1), &
         wkl(mxmol,nl+1), wx(maxxsec,nl+1), semiss(nbndlw), cldfmc(ngptlw,nl+1), taucmc(ngptlw,nl+1), &
         ciwpmc(ngptlw,nl+1), clwpmc(ngptlw,nl+1), cswpmc(ngptlw,nl+1), reicmc(nl+1), relqmc(nl+1), &
         resnmc(nl+1), taua(nl+1,nbndlw), planklay(nl+1,nbndlw), planklev(0:nl+1,nbndlw), &
         plankbnd(nbndlw), jp(nl+1), jt(nl+1), jt1(nl+1), indself(nl+1), indfor(nl+1), indminor(nl+1))
    allocate(colh2o(nl+1), colco2(nl+1), colo3(nl+1), coln2o(nl+1), colco(nl+1), colch4(nl+1), &
         colo2(nl+1), colbrd(nl+1), fac00(nl+1), fac01(nl+1), fac10(nl+1), fac11(nl+1), &
         rat_h2oco2(nl+1), rat_h2oco2_1(nl+1), rat_h2oo3(nl+1), rat_h2oo3_1(nl+1), rat_h2on2o(nl+1), &
         rat_h2on2o_1(nl+1), rat_h2och4(nl+1), rat_h2och4_1(nl+1), rat_n2oco2(nl+1), rat_n2oco2_1(nl+1), &
         rat_o3co2(nl+1), rat_o3co2_1(nl+1), selffac(nl+1), selffrac(nl+1), forfac(nl+1), forfrac(nl+1), &
         minorfrac(nl+1), scaleminor(nl+1), scaleminorn2(nl+1), fracs(nl+1,ngptlw), taug(nl+1,ngptlw), &
         totuflux(0:nl+1), totdflux(0:nl+1), fnet(0:nl+1), htr(0:nl+1), totuclfl(0:nl+1), &
         totdclfl(0:nl+1), fnetc(0:nl+1), htrc(0:nl+1))
    ! dump layouts (Fortran order == Python [L, C, ...] reversed)
    allocate(o_coldry(ncol,nl), o_wbrodl(ncol,nl), o_wkl(7,ncol,nl), o_wx(4,ncol,nl), o_pwvcm(ncol), &
         o_taucmc(ngptlw,ncol,nl), o_lay(37,ncol,nl), o_ilay(6,ncol,nl), o_laytrop(ncol), &
         o_planklay(nbndlw,ncol,nl), o_planklev(nbndlw,ncol,0:nl), o_plankbnd(nbndlw,ncol), &
         o_fracs(ngptlw,ncol,nl), o_taug(ngptlw,ncol,nl), o_flx(8,ncol,0:nl))
    do ic = 1, ncol
      call inatm(ic, nl, 2, 10, play, plev, tlay, tlev, tsfc, h2ovmr, &
           o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr, cfc12vmr, &
           cfc22vmr, ccl4vmr, emis, 2, 3, 1, &
           cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, tauaer, &
           nlayers, pavel, pz, tavel, tz, tbound, semiss, coldry, &
           wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, &
           cldfmc, taucmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, taua)
      o_coldry(ic,:) = coldry(1:nl); o_wbrodl(ic,:) = wbrodl(1:nl)
      o_wkl(:,ic,:) = wkl(1:7,1:nl); o_wx(:,ic,:) = wx(1:4,1:nl); o_pwvcm(ic) = pwvcm
      call cldprmc(nlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc, &
           clwpmc, cswpmc, reicmc, relqmc, resnmc, ncbands, taucmc)
      o_taucmc(:,ic,:) = taucmc(:,1:nl)
      call setcoef(nlayers, 1, pavel, tavel, tz, tbound, semiss, &
           coldry, wkl, wbrodl, &
           laytrop, jp, jt, jt1, planklay, planklev, plankbnd, &
           colh2o, colco2, colo3, coln2o, colco, colch4, colo2, &
           colbrd, fac00, fac01, fac10, fac11, &
           rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
           rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
           rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
           selffac, selffrac, indself, forfac, forfrac, indfor, &
           minorfrac, scaleminor, scaleminorn2, indminor)
      o_laytrop(ic) = laytrop
      o_ilay(1,ic,:) = jp(1:nl); o_ilay(2,ic,:) = jt(1:nl); o_ilay(3,ic,:) = jt1(1:nl)
      o_ilay(4,ic,:) = indself(1:nl); o_ilay(5,ic,:) = indfor(1:nl); o_ilay(6,ic,:) = indminor(1:nl)
      o_lay(1,ic,:) = colh2o(1:nl); o_lay(2,ic,:) = colco2(1:nl); o_lay(3,ic,:) = colo3(1:nl)
      o_lay(4,ic,:) = coln2o(1:nl); o_lay(5,ic,:) = colco(1:nl); o_lay(6,ic,:) = colch4(1:nl)
      o_lay(7,ic,:) = colo2(1:nl); o_lay(8,ic,:) = colbrd(1:nl); o_lay(9,ic,:) = fac00(1:nl)
      o_lay(10,ic,:) = fac01(1:nl); o_lay(11,ic,:) = fac10(1:nl); o_lay(12,ic,:) = fac11(1:nl)
      o_lay(13,ic,:) = rat_h2oco2(1:nl); o_lay(14,ic,:) = rat_h2oco2_1(1:nl)
      o_lay(15,ic,:) = rat_h2oo3(1:nl); o_lay(16,ic,:) = rat_h2oo3_1(1:nl)
      o_lay(17,ic,:) = rat_h2on2o(1:nl); o_lay(18,ic,:) = rat_h2on2o_1(1:nl)
      o_lay(19,ic,:) = rat_h2och4(1:nl); o_lay(20,ic,:) = rat_h2och4_1(1:nl)
      o_lay(21,ic,:) = rat_n2oco2(1:nl); o_lay(22,ic,:) = rat_n2oco2_1(1:nl)
      o_lay(23,ic,:) = rat_o3co2(1:nl); o_lay(24,ic,:) = rat_o3co2_1(1:nl)
      o_lay(25,ic,:) = selffac(1:nl); o_lay(26,ic,:) = selffrac(1:nl)
      o_lay(27,ic,:) = forfac(1:nl); o_lay(28,ic,:) = forfrac(1:nl)
      o_lay(29,ic,:) = minorfrac(1:nl); o_lay(30,ic,:) = scaleminor(1:nl)
      o_lay(31,ic,:) = scaleminorn2(1:nl)
      o_lay(32,ic,:) = pavel(1:nl); o_lay(33,ic,:) = tavel(1:nl)
      o_lay(34,ic,:) = pz(1:nl); o_lay(35,ic,:) = tz(1:nl)
      o_lay(36,ic,:) = cldfmc(1,1:nl); o_lay(37,ic,:) = reicmc(1:nl)
      o_planklay(:,ic,:) = transpose(planklay(1:nl,:))
      o_planklev(:,ic,:) = transpose(planklev(0:nl,:))
      o_plankbnd(:,ic) = plankbnd
      call taumol(nlayers, pavel, wx, coldry, &
           laytrop, jp, jt, jt1, planklay, planklev, plankbnd, &
           colh2o, colco2, colo3, coln2o, colco, colch4, colo2, &
           colbrd, fac00, fac01, fac10, fac11, &
           rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
           rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
           rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
           selffac, selffrac, indself, forfac, forfrac, indfor, &
           minorfrac, scaleminor, scaleminorn2, indminor, &
           fracs, taug)
      o_fracs(:,ic,:) = transpose(fracs(1:nl,:)); o_taug(:,ic,:) = transpose(taug(1:nl,:))
      call rtrnmc(nlayers, 1, 16, 0, pz, semiss, ncbands, &
           cldfmc, taucmc, planklay, planklev, plankbnd, &
           pwvcm, fracs, taug, &
           totuflux, totdflux, fnet, htr, &
           totuclfl, totdclfl, fnetc, htrc)
      o_flx(1,ic,:) = totuflux(0:nl); o_flx(2,ic,:) = totdflux(0:nl); o_flx(3,ic,:) = fnet(0:nl)
      o_flx(4,ic,:) = htr(0:nl); o_flx(5,ic,:) = totuclfl(0:nl); o_flx(6,ic,:) = totdclfl(0:nl)
      o_flx(7,ic,:) = fnetc(0:nl); o_flx(8,ic,:) = htrc(0:nl)
    end do
    call wr('coldry', o_coldry, size(o_coldry)); call wr('wbrodl', o_wbrodl, size(o_wbrodl))
    call wr('wkl', o_wkl, size(o_wkl)); call wr('wx', o_wx, size(o_wx)); call wr('pwvcm', o_pwvcm, size(o_pwvcm))
    call wr('taucmc', o_taucmc, size(o_taucmc)); call wr('lay', o_lay, size(o_lay))
    call wr('ilay', o_ilay, size(o_ilay)); call wr('laytrop', o_laytrop, size(o_laytrop))
    call wr('planklay', o_planklay, size(o_planklay)); call wr('planklev', o_planklev, size(o_planklev))
    call wr('plankbnd', o_plankbnd, size(o_plankbnd)); call wr('fracs', o_fracs, size(o_fracs))
    call wr('taug', o_taug, size(o_taug)); call wr('flx', o_flx, size(o_flx))
  end subroutine chain

  subroutine lwrad
    integer :: nk
    nk = nlay
    allocate(p8w(ncol,1,nk+1), p3d(ncol,1,nk+1), pi3d(ncol,1,nk+1), dz8w(ncol,1,nk+1), &
         t3d(ncol,1,nk+1), t8w(ncol,1,nk+1), rho3d(ncol,1,nk+1), cldfra3d(ncol,1,nk+1), &
         qv3d(ncol,1,nk+1), qc3d(ncol,1,nk+1), qr3d(ncol,1,nk+1), qi3d(ncol,1,nk+1), &
         qs3d(ncol,1,nk+1), qg3d(ncol,1,nk+1), o33d(ncol,1,nk+1), rthratenlw(ncol,1,nk+1), &
         lwtenc(ncol,1,nk+1), re_cloud(ncol,1,nk+1), re_ice(ncol,1,nk+1), re_snow(ncol,1,nk+1))
    allocate(tsk(ncol,1), emiss(ncol,1), xland(ncol,1), xice(ncol,1), snow(ncol,1), glw(ncol,1), &
         olr(ncol,1), lwcf(ncol,1), lwupt(ncol,1), lwuptc(ncol,1), lwdnt(ncol,1), lwdntc(ncol,1), &
         lwupb(ncol,1), lwupbc(ncol,1), lwdnb(ncol,1), lwdnbc(ncol,1), lwuptcln(ncol,1), &
         lwdntcln(ncol,1), lwupbcln(ncol,1), lwdnbcln(ncol,1))
    p8w = 0.0; p3d = 0.0; t3d = 0.0; t8w = 0.0; cldfra3d = 0.0; qv3d = 0.0; qc3d = 0.0
    qr3d = 0.0; qi3d = 0.0; qs3d = 0.0; qg3d = 0.0; o33d = 0.0; rthratenlw = 0.0; lwtenc = 0.0
    pi3d = 1.0; dz8w = 100.0; rho3d = 1.0; re_cloud = 0.0; re_ice = 0.0; re_snow = 0.0
    call rd('p8w', p8w, size(p8w)); call rd('p3d', p3d, size(p3d)); call rd('t3d', t3d, size(t3d))
    call rd('t8w', t8w, size(t8w)); call rd('cldfra3d', cldfra3d, size(cldfra3d))
    call rd('qv3d', qv3d, size(qv3d)); call rd('qc3d', qc3d, size(qc3d)); call rd('qr3d', qr3d, size(qr3d))
    call rd('tsk', tsk, size(tsk)); call rd('emiss', emiss, size(emiss)); call rd('xland', xland, size(xland))
    xice = 0.0; snow = 0.0; glw = 0.0; olr = 0.0; lwcf = 0.0
    julian = 134.0
    call RRTMG_LWRAD(                                      &
         RTHRATENLW=rthratenlw,lwtenc=lwtenc,              &
         LWUPT=LWUPT,LWUPTC=LWUPTC,                        &
         LWDNT=LWDNT,LWDNTC=LWDNTC,                        &
         LWUPB=LWUPB,LWUPBC=LWUPBC,                        &
         LWDNB=LWDNB,LWDNBC=LWDNBC,                        &
         GLW=GLW,OLR=olr,LWCF=LWCF,                        &
         EMISS=EMISS,                                      &
         P8W=p8w,P3D=p3d,PI3D=pi3d,DZ8W=dz8w,TSK=tsk,T3D=t3d, &
         T8W=t8w,RHO3D=rho3d,R=287.04,G=9.81,              &
         ICLOUD=1,WARM_RAIN=.false.,                       &
         CLDFRA3D=cldfra3d,                                &
         cldovrlp=2,                                       &
         IS_CAMMGMP_USED=.false.,                          &
         XLAND=XLAND,XICE=XICE,SNOW=snow,                  &
         QV3D=qv3d,QC3D=qc3d,QR3D=qr3d,                    &
         QI3D=qi3d,QS3D=qs3d,QG3D=qg3d,                    &
         O3INPUT=0,O33D=o33d,                              &
         F_QV=.true.,F_QC=.true.,F_QR=.true.,              &
         F_QI=.false.,F_QS=.false.,F_QG=.false.,           &
         RE_CLOUD=re_cloud,RE_ICE=re_ice,RE_SNOW=re_snow,  &
         has_reqc=0,has_reqi=0,has_reqs=0,                 &
         YR=yr,JULIAN=julian,                              &
         calc_clean_atm_diag=0,                            &
         ids=1  ,ide=ncol+1 , jds= 1 ,jde=2 , kds=1  ,kde=nk+1 ,  &
         ims=1 ,ime=ncol   , jms=1 ,jme=1   , kms=1 ,kme=nk+1 ,   &
         its=1  ,ite=ncol   , jts=1  ,jte=1   , kts=1  ,kte=nk ,  &
         mp_physics=1,co2out=co2out               )
    call wr('rthratenlw', rthratenlw, size(rthratenlw)); call wr('lwtenc', lwtenc, size(lwtenc))
    call wr('o33d', o33d, size(o33d)); call wr('glw', glw, size(glw)); call wr('olr', olr, size(olr))
    call wr('lwcf', lwcf, size(lwcf)); call wr('lwupt', lwupt, size(lwupt)); call wr('lwuptc', lwuptc, size(lwuptc))
    call wr('lwdnt', lwdnt, size(lwdnt)); call wr('lwdntc', lwdntc, size(lwdntc))
    call wr('lwupb', lwupb, size(lwupb)); call wr('lwupbc', lwupbc, size(lwupbc))
    call wr('lwdnb', lwdnb, size(lwdnb)); call wr('lwdnbc', lwdnbc, size(lwdnbc))
    b1(1) = co2out; call wr('co2out', b1, 1)
  end subroutine lwrad
end program drv_rrtmg_lw
