! Harness driver for src/module_ra_rrtmg_sw.F (RRTMG shortwave as CM1 uses it).
! Needs RRTMG_SW_DATA in the working directory.
! params = [mode, ncol, nlay, scon, yr, permuteseed, prmuz]
!   mode 1: rrtmg_swinit, dump the reduced tables (b<NN>_<name>)
!   mode 2: mcica_subcol_sw + the rrtmg_sw chain (inatm_sw, cldprmc_sw,
!           setcoef_sw, taumol_sw, spcvmc_sw) per column, dumping every
!           intermediate, and rrtmg_sw itself on all columns
!   mode 3: RRTMG_SWRAD on ncol columns (nk = nlay model layers)
!   mode 4: reftra_sw on nlay layers (prmuz = params(7))
program drv_rrtmg_sw
  use harness_io
  use parrrsw, only : nbndsw, ngptsw, mxmol, jpband
  use rrsw_tbl, only : exp_tbl
  use rrsw_wvn, only : rwgt
  use rrsw_cld, only : extliq1, ssaliq1, asyliq1, extice3, ssaice3, asyice3, fdlice3
  use rrsw_ref, only : pref, preflog, tref
  use rrtmg_sw_rad, only : rrtmg_sw, inatm_sw
  use rrtmg_sw_cldprmc, only : cldprmc_sw
  use rrtmg_sw_setcoef, only : setcoef_sw
  use rrtmg_sw_taumol, only : taumol_sw
  use rrtmg_sw_spcvmc, only : spcvmc_sw
  use rrtmg_sw_reftra, only : reftra_sw
  use mcica_subcol_gen_sw, only : mcica_subcol_sw
  use module_ra_rrtmg_sw, only : rrtmg_swinit, rrtmg_swrad
  implicit none
  real :: params(7), scon, prmuz, b1(1)
  integer :: mode, ncol, nlay, yr, permuteseed

  call rd('params', params, size(params))
  mode = nint(params(1)); ncol = nint(params(2)); nlay = nint(params(3))
  scon = params(4); yr = nint(params(5)); permuteseed = nint(params(6)); prmuz = params(7)
  call rrtmg_swinit(.true., 0, 1, ncol+1, 1, 2, 1, nlay+1, &
       1, ncol, 1, 1, 1, nlay+1, 1, ncol, 1, 1, 1, nlay)

  select case (mode)
  case (1)
    call dump16; call dump17; call dump18; call dump19; call dump20; call dump21; call dump22
    call dump23; call dump24; call dump25; call dump26; call dump27; call dump28; call dump29
    call wr('exp_tbl', exp_tbl, size(exp_tbl)); call wr('rwgt', rwgt, size(rwgt))
    call wr('extliq1', extliq1, size(extliq1)); call wr('ssaliq1', ssaliq1, size(ssaliq1))
    call wr('asyliq1', asyliq1, size(asyliq1)); call wr('extice3', extice3, size(extice3))
    call wr('ssaice3', ssaice3, size(ssaice3)); call wr('asyice3', asyice3, size(asyice3))
    call wr('fdlice3', fdlice3, size(fdlice3)); call wr('pref', pref, size(pref))
    call wr('preflog', preflog, size(preflog)); call wr('tref', tref, size(tref))
  case (2)
    call chain
  case (3)
    call swrad
  case (4)
    call reftra
  end select

contains
  subroutine dump16
    use rrsw_kg16, only : rayl, strrat1, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b16_rayl', b1, 1)
    b1(1) = strrat1; call wr('b16_strrat1', b1, 1)
    b1(1) = layreffr; call wr('b16_layreffr', b1, 1)
    call wr('b16_selfref', selfref, size(selfref))
    call wr('b16_forref', forref, size(forref))
    call wr('b16_sfluxref', sfluxref, size(sfluxref))
    call wr('b16_absa', absa, size(absa))
    call wr('b16_absb', absb, size(absb))
  end subroutine dump16
  subroutine dump17
    use rrsw_kg17, only : rayl, strrat, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b17_rayl', b1, 1)
    b1(1) = strrat; call wr('b17_strrat', b1, 1)
    b1(1) = layreffr; call wr('b17_layreffr', b1, 1)
    call wr('b17_selfref', selfref, size(selfref))
    call wr('b17_forref', forref, size(forref))
    call wr('b17_sfluxref', sfluxref, size(sfluxref))
    call wr('b17_absa', absa, size(absa))
    call wr('b17_absb', absb, size(absb))
  end subroutine dump17
  subroutine dump18
    use rrsw_kg18, only : rayl, strrat, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b18_rayl', b1, 1)
    b1(1) = strrat; call wr('b18_strrat', b1, 1)
    b1(1) = layreffr; call wr('b18_layreffr', b1, 1)
    call wr('b18_selfref', selfref, size(selfref))
    call wr('b18_forref', forref, size(forref))
    call wr('b18_sfluxref', sfluxref, size(sfluxref))
    call wr('b18_absa', absa, size(absa))
    call wr('b18_absb', absb, size(absb))
  end subroutine dump18
  subroutine dump19
    use rrsw_kg19, only : rayl, strrat, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b19_rayl', b1, 1)
    b1(1) = strrat; call wr('b19_strrat', b1, 1)
    b1(1) = layreffr; call wr('b19_layreffr', b1, 1)
    call wr('b19_selfref', selfref, size(selfref))
    call wr('b19_forref', forref, size(forref))
    call wr('b19_sfluxref', sfluxref, size(sfluxref))
    call wr('b19_absa', absa, size(absa))
    call wr('b19_absb', absb, size(absb))
  end subroutine dump19
  subroutine dump20
    use rrsw_kg20, only : rayl, layreffr, absch4, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b20_rayl', b1, 1)
    b1(1) = layreffr; call wr('b20_layreffr', b1, 1)
    call wr('b20_absch4', absch4, size(absch4))
    call wr('b20_selfref', selfref, size(selfref))
    call wr('b20_forref', forref, size(forref))
    call wr('b20_sfluxref', sfluxref, size(sfluxref))
    call wr('b20_absa', absa, size(absa))
    call wr('b20_absb', absb, size(absb))
  end subroutine dump20
  subroutine dump21
    use rrsw_kg21, only : rayl, strrat, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b21_rayl', b1, 1)
    b1(1) = strrat; call wr('b21_strrat', b1, 1)
    b1(1) = layreffr; call wr('b21_layreffr', b1, 1)
    call wr('b21_selfref', selfref, size(selfref))
    call wr('b21_forref', forref, size(forref))
    call wr('b21_sfluxref', sfluxref, size(sfluxref))
    call wr('b21_absa', absa, size(absa))
    call wr('b21_absb', absb, size(absb))
  end subroutine dump21
  subroutine dump22
    use rrsw_kg22, only : rayl, strrat, layreffr, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b22_rayl', b1, 1)
    b1(1) = strrat; call wr('b22_strrat', b1, 1)
    b1(1) = layreffr; call wr('b22_layreffr', b1, 1)
    call wr('b22_selfref', selfref, size(selfref))
    call wr('b22_forref', forref, size(forref))
    call wr('b22_sfluxref', sfluxref, size(sfluxref))
    call wr('b22_absa', absa, size(absa))
    call wr('b22_absb', absb, size(absb))
  end subroutine dump22
  subroutine dump23
    use rrsw_kg23, only : rayl, givfac, layreffr, selfref, forref, sfluxref, absa
    call wr('b23_rayl', rayl, size(rayl))
    b1(1) = givfac; call wr('b23_givfac', b1, 1)
    b1(1) = layreffr; call wr('b23_layreffr', b1, 1)
    call wr('b23_selfref', selfref, size(selfref))
    call wr('b23_forref', forref, size(forref))
    call wr('b23_sfluxref', sfluxref, size(sfluxref))
    call wr('b23_absa', absa, size(absa))
  end subroutine dump23
  subroutine dump24
    use rrsw_kg24, only : rayla, raylb, strrat, layreffr, abso3a, abso3b, selfref, forref, sfluxref, absa, absb
    call wr('b24_rayla', rayla, size(rayla))
    call wr('b24_raylb', raylb, size(raylb))
    b1(1) = strrat; call wr('b24_strrat', b1, 1)
    b1(1) = layreffr; call wr('b24_layreffr', b1, 1)
    call wr('b24_abso3a', abso3a, size(abso3a))
    call wr('b24_abso3b', abso3b, size(abso3b))
    call wr('b24_selfref', selfref, size(selfref))
    call wr('b24_forref', forref, size(forref))
    call wr('b24_sfluxref', sfluxref, size(sfluxref))
    call wr('b24_absa', absa, size(absa))
    call wr('b24_absb', absb, size(absb))
  end subroutine dump24
  subroutine dump25
    use rrsw_kg25, only : rayl, layreffr, abso3a, abso3b, sfluxref, absa
    call wr('b25_rayl', rayl, size(rayl))
    b1(1) = layreffr; call wr('b25_layreffr', b1, 1)
    call wr('b25_abso3a', abso3a, size(abso3a))
    call wr('b25_abso3b', abso3b, size(abso3b))
    call wr('b25_sfluxref', sfluxref, size(sfluxref))
    call wr('b25_absa', absa, size(absa))
  end subroutine dump25
  subroutine dump26
    use rrsw_kg26, only : rayl, sfluxref
    call wr('b26_rayl', rayl, size(rayl))
    call wr('b26_sfluxref', sfluxref, size(sfluxref))
  end subroutine dump26
  subroutine dump27
    use rrsw_kg27, only : rayl, scalekur, layreffr, sfluxref, absa, absb
    call wr('b27_rayl', rayl, size(rayl))
    b1(1) = scalekur; call wr('b27_scalekur', b1, 1)
    b1(1) = layreffr; call wr('b27_layreffr', b1, 1)
    call wr('b27_sfluxref', sfluxref, size(sfluxref))
    call wr('b27_absa', absa, size(absa))
    call wr('b27_absb', absb, size(absb))
  end subroutine dump27
  subroutine dump28
    use rrsw_kg28, only : rayl, strrat, layreffr, sfluxref, absa, absb
    b1(1) = rayl; call wr('b28_rayl', b1, 1)
    b1(1) = strrat; call wr('b28_strrat', b1, 1)
    b1(1) = layreffr; call wr('b28_layreffr', b1, 1)
    call wr('b28_sfluxref', sfluxref, size(sfluxref))
    call wr('b28_absa', absa, size(absa))
    call wr('b28_absb', absb, size(absb))
  end subroutine dump28
  subroutine dump29
    use rrsw_kg29, only : rayl, layreffr, absh2o, absco2, selfref, forref, sfluxref, absa, absb
    b1(1) = rayl; call wr('b29_rayl', b1, 1)
    b1(1) = layreffr; call wr('b29_layreffr', b1, 1)
    call wr('b29_absh2o', absh2o, size(absh2o))
    call wr('b29_absco2', absco2, size(absco2))
    call wr('b29_selfref', selfref, size(selfref))
    call wr('b29_forref', forref, size(forref))
    call wr('b29_sfluxref', sfluxref, size(sfluxref))
    call wr('b29_absa', absa, size(absa))
    call wr('b29_absb', absb, size(absb))
  end subroutine dump29
  subroutine reftra
    real, allocatable :: chk(:), pgg(:), ptau(:), pw(:), pref_(:), prefd(:), ptra(:), ptrad(:)
    logical, allocatable :: lrtchk(:)
    allocate(chk(nlay), pgg(nlay), ptau(nlay), pw(nlay), pref_(nlay), prefd(nlay), ptra(nlay), &
         ptrad(nlay), lrtchk(nlay))
    call rd('lrtchk', chk, nlay); call rd('pgg', pgg, nlay); call rd('ptau', ptau, nlay)
    call rd('pw', pw, nlay)
    lrtchk = chk > 0.5
    call reftra_sw(nlay, lrtchk, pgg, prmuz, ptau, pw, pref_, prefd, ptra, ptrad)
    call wr('pref', pref_, nlay); call wr('prefd', prefd, nlay)
    call wr('ptra', ptra, nlay); call wr('ptrad', ptrad, nlay)
  end subroutine reftra

  subroutine chain
    integer :: nl, ic, icld, inflag, iceflag, liqflag, nlayers, laytrop, layswtch, laylow, irng, k
    real :: tbound, cossza
    real, allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:), h2ovmr(:,:), &
         o3vmr(:,:), co2vmr(:,:), ch4vmr(:,:), n2ovmr(:,:), o2vmr(:,:), cldfrac(:,:), &
         ciwp(:,:), clwp(:,:), cswp(:,:), rei(:,:), rel(:,:), res(:,:), hgt(:,:), &
         taucld(:,:,:), ssacld(:,:,:), asmcld(:,:,:), fsfcld(:,:,:), &
         asdir(:), asdif(:), aldir(:), aldif(:), coszen(:), &
         tauaer(:,:,:), ssaaer(:,:,:), asmaer(:,:,:), ecaer(:,:,:)
    real, allocatable :: cldfmcl(:,:,:), ciwpmcl(:,:,:), clwpmcl(:,:,:), cswpmcl(:,:,:), &
         taucmcl(:,:,:), ssacmcl(:,:,:), asmcmcl(:,:,:), fsfcmcl(:,:,:), reicmcl(:,:), &
         relqmcl(:,:), resnmcl(:,:)
    real, allocatable :: swuflx(:,:), swdflx(:,:), swhr(:,:), swuflxc(:,:), swdflxc(:,:), &
         swhrc(:,:), swuflxcln(:,:), swdflxcln(:,:), sibvisdir(:,:), sibvisdif(:,:), &
         sibnirdir(:,:), sibnirdif(:,:), swdkdir(:,:), swdkdif(:,:), swdkdirc(:,:)
    ! per-column chain (sizes as in rrtmg_sw)
    real, allocatable :: pavel(:), tavel(:), pz(:), tz(:), pdp(:), coldry(:), wkl(:,:), &
         adjflux(:), solvar(:), taua(:,:), ssaa(:,:), asma(:,:), cldfmc(:,:), taucmc(:,:), &
         ssacmc(:,:), asmcmc(:,:), fsfcmc(:,:), ciwpmc(:,:), clwpmc(:,:), cswpmc(:,:), &
         reicmc(:), relqmc(:), resnmc(:), taormc(:,:), &
         colh2o(:), colco2(:), colo3(:), coln2o(:), colch4(:), colo2(:), colmol(:), co2mult(:), &
         fac00(:), fac01(:), fac10(:), fac11(:), selffac(:), selffrac(:), forfac(:), forfrac(:), &
         sfluxzen(:), taug(:,:), taur(:,:), albdir(:), albdif(:), &
         zcldfmc(:,:), ztaucmc(:,:), ztaormc(:,:), zasycmc(:,:), zomgcmc(:,:), &
         ztaua(:,:), zasya(:,:), zomga(:,:), &
         zbbfd(:), zbbfu(:), zbbcd(:), zbbcu(:), zuvfd(:), zuvcd(:), znifd(:), znicd(:), &
         zbbfddir(:), zbbcddir(:), zuvfddir(:), zuvcddir(:), znifddir(:), znicddir(:)
    integer, allocatable :: jp(:), jt(:), jt1(:), indself(:), indfor(:)
    real, allocatable :: o_coldry(:,:), o_wkl(:,:,:), o_pdp(:,:), o_adj(:), o_cp(:,:,:,:), &
         o_lay(:,:,:), o_ilay(:,:,:), o_lt(:,:), o_sfl(:,:), o_taug(:,:,:), o_taur(:,:,:), &
         o_flx(:,:,:)
    nl = nlay
    allocate(play(ncol,nl), plev(ncol,nl+1), tlay(ncol,nl), tlev(ncol,nl+1), tsfc(ncol), &
         h2ovmr(ncol,nl), o3vmr(ncol,nl), co2vmr(ncol,nl), ch4vmr(ncol,nl), n2ovmr(ncol,nl), &
         o2vmr(ncol,nl), cldfrac(ncol,nl), ciwp(ncol,nl), clwp(ncol,nl), cswp(ncol,nl), &
         rei(ncol,nl), rel(ncol,nl), res(ncol,nl), hgt(ncol,nl), taucld(nbndsw,ncol,nl), &
         ssacld(nbndsw,ncol,nl), asmcld(nbndsw,ncol,nl), fsfcld(nbndsw,ncol,nl), &
         asdir(ncol), asdif(ncol), aldir(ncol), aldif(ncol), coszen(ncol), &
         tauaer(ncol,nl,nbndsw), ssaaer(ncol,nl,nbndsw), asmaer(ncol,nl,nbndsw), ecaer(ncol,nl,6))
    allocate(cldfmcl(ngptsw,ncol,nl), ciwpmcl(ngptsw,ncol,nl), clwpmcl(ngptsw,ncol,nl), &
         cswpmcl(ngptsw,ncol,nl), taucmcl(ngptsw,ncol,nl), ssacmcl(ngptsw,ncol,nl), &
         asmcmcl(ngptsw,ncol,nl), fsfcmcl(ngptsw,ncol,nl), reicmcl(ncol,nl), relqmcl(ncol,nl), &
         resnmcl(ncol,nl))
    allocate(swuflx(ncol,nl+1), swdflx(ncol,nl+1), swhr(ncol,nl), swuflxc(ncol,nl+1), &
         swdflxc(ncol,nl+1), swhrc(ncol,nl), swuflxcln(ncol,nl+1), swdflxcln(ncol,nl+1), &
         sibvisdir(ncol,nl+1), sibvisdif(ncol,nl+1), sibnirdir(ncol,nl+1), sibnirdif(ncol,nl+1), &
         swdkdir(ncol,nl+1), swdkdif(ncol,nl+1), swdkdirc(ncol,nl+1))
    call rd('play', play, size(play)); call rd('plev', plev, size(plev))
    call rd('tlay', tlay, size(tlay)); call rd('tlev', tlev, size(tlev))
    call rd('tsfc', tsfc, size(tsfc)); call rd('h2ovmr', h2ovmr, size(h2ovmr))
    call rd('o3vmr', o3vmr, size(o3vmr)); call rd('co2vmr', co2vmr, size(co2vmr))
    call rd('ch4vmr', ch4vmr, size(ch4vmr)); call rd('n2ovmr', n2ovmr, size(n2ovmr))
    call rd('o2vmr', o2vmr, size(o2vmr))
    call rd('cldfrac', cldfrac, size(cldfrac)); call rd('ciwp', ciwp, size(ciwp))
    call rd('clwp', clwp, size(clwp)); call rd('rei', rei, size(rei)); call rd('rel', rel, size(rel))
    call rd('asdir', asdir, ncol); call rd('asdif', asdif, ncol)
    call rd('aldir', aldir, ncol); call rd('aldif', aldif, ncol); call rd('coszen', coszen, ncol)
    cswp = 0.0; res = 10.0; hgt = 0.0; taucld = 0.0; ssacld = 1.0; asmcld = 0.0; fsfcld = 0.0
    tauaer = 0.0; ssaaer = 1.0; asmaer = 0.0; ecaer = 0.0
    icld = 2; irng = 0
    call mcica_subcol_sw(1, ncol, nl, icld, permuteseed, irng, play, hgt, &
         cldfrac, ciwp, clwp, cswp, rei, rel, res, taucld, ssacld, asmcld, fsfcld, &
         cldfmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, &
         taucmcl, ssacmcl, asmcmcl, fsfcmcl)
    call wr('cldfmcl', cldfmcl, size(cldfmcl)); call wr('ciwpmcl', ciwpmcl, size(ciwpmcl))
    call wr('clwpmcl', clwpmcl, size(clwpmcl)); call wr('taucmcl', taucmcl, size(taucmcl))
    call wr('ssacmcl', ssacmcl, size(ssacmcl)); call wr('asmcmcl', asmcmcl, size(asmcmcl))
    call wr('fsfcmcl', fsfcmcl, size(fsfcmcl))
    call rrtmg_sw(ncol, nl, icld, play, plev, tlay, tlev, tsfc, &
         h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
         asdir, asdif, aldir, aldif, coszen, 1.0, 0, scon, &
         2, 3, 1, cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl, &
         ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, &
         tauaer, ssaaer, asmaer, ecaer, &
         swuflx, swdflx, swhr, swuflxc, swdflxc, swhrc, swuflxcln, swdflxcln, 0, &
         sibvisdir, sibvisdif, sibnirdir, sibnirdif, swdkdir, swdkdif, swdkdirc, 0)
    call wr('swuflx', swuflx, size(swuflx)); call wr('swdflx', swdflx, size(swdflx))
    call wr('swuflxc', swuflxc, size(swuflxc)); call wr('swdflxc', swdflxc, size(swdflxc))
    call wr('swhr', swhr, size(swhr)); call wr('swhrc', swhrc, size(swhrc))
    call wr('swdkdir', swdkdir, size(swdkdir)); call wr('swdkdif', swdkdif, size(swdkdif))
    call wr('swdkdirc', swdkdirc, size(swdkdirc))
    ! the chain, one column at a time
    allocate(pavel(nl+1), tavel(nl+1), pz(0:nl+1), tz(0:nl+1), pdp(nl+1), coldry(nl+1), &
         wkl(mxmol,nl+1), adjflux(jpband), solvar(jpband), taua(nl+1,nbndsw), &
         ssaa(nl+1,nbndsw), asma(nl+1,nbndsw), cldfmc(ngptsw,nl+1), taucmc(ngptsw,nl+1), &
         ssacmc(ngptsw,nl+1), asmcmc(ngptsw,nl+1), fsfcmc(ngptsw,nl+1), ciwpmc(ngptsw,nl+1), &
         clwpmc(ngptsw,nl+1), cswpmc(ngptsw,nl+1), reicmc(nl+1), relqmc(nl+1), resnmc(nl+1), &
         taormc(ngptsw,nl+1))
    allocate(colh2o(nl+1), colco2(nl+1), colo3(nl+1), coln2o(nl+1), colch4(nl+1), colo2(nl+1), &
         colmol(nl+1), co2mult(nl+1), fac00(nl+1), fac01(nl+1), fac10(nl+1), fac11(nl+1), &
         selffac(nl+1), selffrac(nl+1), forfac(nl+1), forfrac(nl+1), jp(nl+1), jt(nl+1), &
         jt1(nl+1), indself(nl+1), indfor(nl+1), sfluxzen(ngptsw), taug(nl+1,ngptsw), &
         taur(nl+1,ngptsw), albdir(nbndsw), albdif(nbndsw), zcldfmc(nl+1,ngptsw), &
         ztaucmc(nl+1,ngptsw), ztaormc(nl+1,ngptsw), zasycmc(nl+1,ngptsw), zomgcmc(nl+1,ngptsw), &
         ztaua(nl+1,nbndsw), zasya(nl+1,nbndsw), zomga(nl+1,nbndsw), &
         zbbfd(nl+2), zbbfu(nl+2), zbbcd(nl+2), zbbcu(nl+2), zuvfd(nl+2), zuvcd(nl+2), &
         znifd(nl+2), znicd(nl+2), zbbfddir(nl+2), zbbcddir(nl+2), zuvfddir(nl+2), &
         zuvcddir(nl+2), znifddir(nl+2), znicddir(nl+2))
    allocate(o_coldry(ncol,nl), o_wkl(7,ncol,nl), o_pdp(ncol,nl), o_adj(ncol), &
         o_cp(ngptsw,ncol,nl,4), o_lay(16,ncol,nl), o_ilay(5,ncol,nl), o_lt(2,ncol), &
         o_sfl(ngptsw,ncol), o_taug(ngptsw,ncol,nl), o_taur(ngptsw,ncol,nl), o_flx(6,ncol,nl+1))
    do ic = 1, ncol
      call inatm_sw(ic, nl, icld, 10, play, plev, tlay, tlev, tsfc, h2ovmr, &
           o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, 1.0, 0, scon, 2, 3, 1, &
           cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl, ciwpmcl, clwpmcl, cswpmcl, &
           reicmcl, relqmcl, resnmcl, tauaer, ssaaer, asmaer, &
           nlayers, pavel, pz, pdp, tavel, tz, tbound, coldry, wkl, &
           adjflux, solvar, inflag, iceflag, liqflag, cldfmc, taucmc, &
           ssacmc, asmcmc, fsfcmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, &
           taua, ssaa, asma)
      o_coldry(ic,:) = coldry(1:nl); o_wkl(:,ic,:) = wkl(1:7,1:nl); o_pdp(ic,:) = pdp(1:nl)
      o_adj(ic) = adjflux(16)
      call cldprmc_sw(nlayers, inflag, iceflag, liqflag, cldfmc, &
           ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, &
           taormc, taucmc, ssacmc, asmcmc, fsfcmc)
      o_cp(:,ic,:,1) = taormc(:,1:nl); o_cp(:,ic,:,2) = taucmc(:,1:nl)
      o_cp(:,ic,:,3) = ssacmc(:,1:nl); o_cp(:,ic,:,4) = asmcmc(:,1:nl)
      call setcoef_sw(nlayers, pavel, tavel, pz, tz, tbound, coldry, wkl, &
           laytrop, layswtch, laylow, jp, jt, jt1, &
           co2mult, colch4, colco2, colh2o, colmol, coln2o, &
           colo2, colo3, fac00, fac01, fac10, fac11, &
           selffac, selffrac, indself, forfac, forfrac, indfor)
      o_lt(1,ic) = laytrop; o_lt(2,ic) = laylow
      o_ilay(1,ic,:) = jp(1:nl); o_ilay(2,ic,:) = jt(1:nl); o_ilay(3,ic,:) = jt1(1:nl)
      o_ilay(4,ic,:) = indself(1:nl); o_ilay(5,ic,:) = indfor(1:nl)
      o_lay(1,ic,:) = colh2o(1:nl); o_lay(2,ic,:) = colco2(1:nl); o_lay(3,ic,:) = colo3(1:nl)
      o_lay(4,ic,:) = coln2o(1:nl); o_lay(5,ic,:) = colch4(1:nl); o_lay(6,ic,:) = colo2(1:nl)
      o_lay(7,ic,:) = colmol(1:nl); o_lay(8,ic,:) = co2mult(1:nl)
      o_lay(9,ic,:) = fac00(1:nl); o_lay(10,ic,:) = fac01(1:nl)
      o_lay(11,ic,:) = fac10(1:nl); o_lay(12,ic,:) = fac11(1:nl)
      o_lay(13,ic,:) = selffac(1:nl); o_lay(14,ic,:) = selffrac(1:nl)
      o_lay(15,ic,:) = forfac(1:nl); o_lay(16,ic,:) = forfrac(1:nl)
      call taumol_sw(nlayers, colh2o, colco2, colch4, colo2, colo3, colmol, &
           laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11, &
           selffac, selffrac, indself, forfac, forfrac, indfor, sfluxzen, taug, taur)
      o_sfl(:,ic) = sfluxzen
      o_taug(:,ic,:) = transpose(taug(1:nl,:)); o_taur(:,ic,:) = transpose(taur(1:nl,:))
      cossza = coszen(ic)
      if (cossza .le. 1.e-10) cossza = 1.e-10
      albdir(1:9) = aldir(ic); albdif(1:9) = aldif(ic)
      albdir(nbndsw) = aldir(ic); albdif(nbndsw) = aldif(ic)
      albdir(10:13) = asdir(ic); albdif(10:13) = asdif(ic)
      do k = 1, nlayers
        zcldfmc(k,:) = cldfmc(:,k); ztaucmc(k,:) = taucmc(:,k); ztaormc(k,:) = taormc(:,k)
        zasycmc(k,:) = asmcmc(:,k); zomgcmc(k,:) = ssacmc(:,k)
      end do
      ztaua = taua; zasya = asma; zomga = ssaa
      zbbfd = 0.; zbbfu = 0.; zbbcd = 0.; zbbcu = 0.; zbbfddir = 0.; zbbcddir = 0.
      call spcvmc_sw(nlayers, 16, 29, 1, 0, &
           pavel, tavel, pz, tz, tbound, albdif, albdir, &
           zcldfmc, ztaucmc, zasycmc, zomgcmc, ztaormc, &
           ztaua, zasya, zomga, cossza, coldry, wkl, adjflux, &
           laytrop, layswtch, laylow, jp, jt, jt1, &
           co2mult, colch4, colco2, colh2o, colmol, coln2o, colo2, colo3, &
           fac00, fac01, fac10, fac11, &
           selffac, selffrac, indself, forfac, forfrac, indfor, &
           zbbfd, zbbfu, zbbcd, zbbcu, zuvfd, zuvcd, znifd, znicd, &
           zbbfddir, zbbcddir, zuvfddir, zuvcddir, znifddir, znicddir)
      o_flx(1,ic,:) = zbbfu(1:nl+1); o_flx(2,ic,:) = zbbfd(1:nl+1)
      o_flx(3,ic,:) = zbbcu(1:nl+1); o_flx(4,ic,:) = zbbcd(1:nl+1)
      o_flx(5,ic,:) = zbbfddir(1:nl+1); o_flx(6,ic,:) = zbbcddir(1:nl+1)
    end do
    call wr('coldry', o_coldry, size(o_coldry)); call wr('wkl', o_wkl, size(o_wkl))
    call wr('pdp', o_pdp, size(o_pdp)); call wr('adjflux', o_adj, size(o_adj))
    call wr('cldprmc', o_cp, size(o_cp)); call wr('lay', o_lay, size(o_lay))
    call wr('ilay', o_ilay, size(o_ilay)); call wr('laytrop', o_lt, size(o_lt))
    call wr('sfluxzen', o_sfl, size(o_sfl)); call wr('taug', o_taug, size(o_taug))
    call wr('taur', o_taur, size(o_taur)); call wr('flx', o_flx, size(o_flx))
  end subroutine chain

  subroutine swrad
    integer :: nk
    real, allocatable :: p8w(:,:,:), p3d(:,:,:), pi3d(:,:,:), dz8w(:,:,:), t3d(:,:,:), &
         t8w(:,:,:), rho3d(:,:,:), cldfra3d(:,:,:), qv3d(:,:,:), qc3d(:,:,:), qr3d(:,:,:), &
         qi3d(:,:,:), qs3d(:,:,:), qg3d(:,:,:), o33d(:,:,:), rthratensw(:,:,:), &
         swtenc(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
    real, allocatable :: tsk(:,:), albedo(:,:), xland(:,:), xice(:,:), snow(:,:), xlat(:,:), &
         xlong(:,:), gsw(:,:), swcf(:,:), coszr(:,:), xcoszen(:,:), swupt(:,:), swuptc(:,:), &
         swuptcln(:,:), swdnt(:,:), swdntc(:,:), swdntcln(:,:), swupb(:,:), swupbc(:,:), &
         swupbcln(:,:), swdnb(:,:), swdnbc(:,:), swdnbcln(:,:), swddir(:,:), swddni(:,:), &
         swddif(:,:), swdownc(:,:), swddnic(:,:), swddirc(:,:)
    real, dimension(:,:,:,:), pointer :: tauaer_sw=>null(), ssaaer_sw=>null(), asyaer_sw=>null()
    real :: co2out, julian
    nk = nlay
    allocate(p8w(ncol,1,nk+1), p3d(ncol,1,nk+1), pi3d(ncol,1,nk+1), dz8w(ncol,1,nk+1), &
         t3d(ncol,1,nk+1), t8w(ncol,1,nk+1), rho3d(ncol,1,nk+1), cldfra3d(ncol,1,nk+1), &
         qv3d(ncol,1,nk+1), qc3d(ncol,1,nk+1), qr3d(ncol,1,nk+1), qi3d(ncol,1,nk+1), &
         qs3d(ncol,1,nk+1), qg3d(ncol,1,nk+1), o33d(ncol,1,nk+1), rthratensw(ncol,1,nk+1), &
         swtenc(ncol,1,nk+1), re_cloud(ncol,1,nk+1), re_ice(ncol,1,nk+1), re_snow(ncol,1,nk+1))
    allocate(tsk(ncol,1), albedo(ncol,1), xland(ncol,1), xice(ncol,1), snow(ncol,1), &
         xlat(ncol,1), xlong(ncol,1), gsw(ncol,1), swcf(ncol,1), coszr(ncol,1), xcoszen(ncol,1), &
         swupt(ncol,1), swuptc(ncol,1), swuptcln(ncol,1), swdnt(ncol,1), swdntc(ncol,1), &
         swdntcln(ncol,1), swupb(ncol,1), swupbc(ncol,1), swupbcln(ncol,1), swdnb(ncol,1), &
         swdnbc(ncol,1), swdnbcln(ncol,1), swddir(ncol,1), swddni(ncol,1), swddif(ncol,1), &
         swdownc(ncol,1), swddnic(ncol,1), swddirc(ncol,1))
    p8w = 0.0; p3d = 0.0; t3d = 0.0; t8w = 0.0; cldfra3d = 0.0; qv3d = 0.0; qc3d = 0.0
    qr3d = 0.0; qi3d = 0.0; qs3d = 0.0; qg3d = 0.0; o33d = 0.0; rthratensw = 0.0
    pi3d = 1.0; dz8w = 0.0; rho3d = 1.0; re_cloud = 0.0; re_ice = 0.0; re_snow = 0.0
    call rd('p8w', p8w, size(p8w)); call rd('p3d', p3d, size(p3d)); call rd('t3d', t3d, size(t3d))
    call rd('t8w', t8w, size(t8w)); call rd('cldfra3d', cldfra3d, size(cldfra3d))
    call rd('dz8w', dz8w, size(dz8w))
    call rd('qv3d', qv3d, size(qv3d)); call rd('qc3d', qc3d, size(qc3d)); call rd('qr3d', qr3d, size(qr3d))
    call rd('tsk', tsk, size(tsk)); call rd('albedo', albedo, size(albedo))
    call rd('xland', xland, size(xland)); call rd('xcoszen', xcoszen, size(xcoszen))
    call rd('swtenc_in', swtenc, size(swtenc)); call rd('gsw_in', gsw, size(gsw))
    xice = 0.0; snow = 0.0; xlat = 0.0; xlong = 0.0; swcf = -1.0; coszr = 0.0
    swupt = -1.0; swuptc = -1.0; swdnt = -1.0; swdntc = -1.0; swupb = -1.0; swupbc = -1.0
    swdnb = -1.0; swdnbc = -1.0
    julian = 134.0
    call RRTMG_SWRAD(                                         &
         RTHRATENSW=rthratensw,swtenc=swtenc,              &
         SWUPT=SWUPT,SWUPTC=SWUPTC,                        &
         SWDNT=SWDNT,SWDNTC=SWDNTC,                        &
         SWUPB=SWUPB,SWUPBC=SWUPBC,                        &
         SWDNB=SWDNB,SWDNBC=SWDNBC,                        &
         SWCF=SWCF,GSW=GSW,                                &
         XTIME=0.0,GMT=0.0,XLAT=XLAT,XLONG=XLONG,          &
         RADT=1.0,DEGRAD=0.017453293,DECLIN=0.0,           &
         COSZR=COSZR,JULDAY=134,SOLCON=scon,               &
         ALBEDO=albedo,t3d=t3d,t8w=t8w,TSK=TSK,            &
         p3d=p3d,p8w=p8w,pi3d=pi3d,rho3d=rho3d,            &
         dz8w=dz8w,                                        &
         CLDFRA3D=cldfra3d,                                &
         IS_CAMMGMP_USED=.false.,                          &
         R=287.04,G=9.81,                                  &
         ICLOUD=1,WARM_RAIN=.false.,                       &
         cldovrlp=2,                                       &
         XLAND=XLAND,XICE=XICE,SNOW=snow,                  &
         QV3D=qv3d,QC3D=qc3d,QR3D=qr3d,                    &
         QI3D=qi3d,QS3D=qs3d,QG3D=qg3d,                    &
         O3INPUT=0,O33D=o33d,                              &
         AER_OPT=0,                                        &
         no_src=6,                                         &
         SF_SURFACE_PHYSICS=1,                             &
         F_QV=.true.,F_QC=.true.,F_QR=.true.,              &
         F_QI=.false.,F_QS=.false.,F_QG=.false.,           &
         RE_CLOUD=re_cloud,RE_ICE=re_ice,RE_SNOW=re_snow,  &
         has_reqc=0,has_reqi=0,has_reqs=0,                 &
         calc_clean_atm_diag=0,                            &
         ids=1  ,ide=ncol+1 , jds= 1 ,jde=2 , kds=1  ,kde=nk+1 ,  &
         ims=1 ,ime=ncol   , jms=1 ,jme=1   , kms=1 ,kme=nk+1 ,   &
         its=1  ,ite=ncol   , jts=1  ,jte=1   , kts=1  ,kte=nk ,  &
         tauaer3d_sw=tauaer_sw,                            &
         ssaaer3d_sw=ssaaer_sw,                            &
         asyaer3d_sw=asyaer_sw,                            &
         swddir=swddir,swddni=swddni,swddif=swddif,        &
         swdownc=swdownc, swddnic=swddnic, swddirc=swddirc, &
         xcoszen=xcoszen,yr=yr,julian=julian,mp_physics=1,co2out=co2out )
    call wr('rthratensw', rthratensw, size(rthratensw)); call wr('swtenc', swtenc, size(swtenc))
    call wr('gsw', gsw, size(gsw)); call wr('swcf', swcf, size(swcf)); call wr('coszr', coszr, size(coszr))
    call wr('swupt', swupt, size(swupt)); call wr('swuptc', swuptc, size(swuptc))
    call wr('swdnt', swdnt, size(swdnt)); call wr('swdntc', swdntc, size(swdntc))
    call wr('swupb', swupb, size(swupb)); call wr('swupbc', swupbc, size(swupbc))
    call wr('swdnb', swdnb, size(swdnb)); call wr('swdnbc', swdnbc, size(swdnbc))
    call wr('swddir', swddir, size(swddir)); call wr('swddni', swddni, size(swddni))
    call wr('swddif', swddif, size(swddif)); call wr('swdownc', swdownc, size(swdownc))
    call wr('swddnic', swddnic, size(swddnic)); call wr('swddirc', swddirc, size(swddirc))
    b1(1) = co2out; call wr('co2out', b1, 1)
  end subroutine swrad
end program drv_rrtmg_sw
