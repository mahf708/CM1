!  (included twice from cm1.F; see cm1dump.F)
        call dump_begin(nstep,mtime)
        call dump1('xh',xh,ib,ie)
        call dump1('rxh',rxh,ib,ie)
        call dump1('uh',uh,ib,ie)
        call dump1('ruh',ruh,ib,ie)
        call dump1('xf',xf,ib,ie+1)
        call dump1('rxf',rxf,ib,ie+1)
        call dump1('uf',uf,ib,ie+1)
        call dump1('ruf',ruf,ib,ie+1)
        call dump1('yh',yh,jb,je)
        call dump1('vh',vh,jb,je)
        call dump1('rvh',rvh,jb,je)
        call dump1('yf',yf,jb,je+1)
        call dump1('vf',vf,jb,je+1)
        call dump1('rvf',rvf,jb,je+1)
        call dump1('sigma',sigma,kb,ke)
        call dump1('rds',rds,kb,ke)
        call dump1('sigmaf',sigmaf,kb,ke+1)
        call dump1('rdsf',rdsf,kb,ke+1)
        call dump3('zh',zh,ib,ie,jb,je,kb,ke)
        call dump3('mh',mh,ib,ie,jb,je,kb,ke)
        call dump3('rmh',rmh,ib,ie,jb,je,kb,ke)
        call dump3('c1',c1,ib,ie,jb,je,kb,ke)
        call dump3('c2',c2,ib,ie,jb,je,kb,ke)
        call dump3('zf',zf,ib,ie,jb,je,kb,ke+1)
        call dump3('mf',mf,ib,ie,jb,je,kb,ke+1)
        call dump3('rmf',rmf,ib,ie,jb,je,kb,ke+1)
        call dump3('pi0',pi0,ib,ie,jb,je,kb,ke)
        call dump3('th0',th0,ib,ie,jb,je,kb,ke)
        call dump3('rho0',rho0,ib,ie,jb,je,kb,ke)
        call dump3('prs0',prs0,ib,ie,jb,je,kb,ke)
        call dump3('thv0',thv0,ib,ie,jb,je,kb,ke)
        call dump3('rth0',rth0,ib,ie,jb,je,kb,ke)
        call dump3('qv0',qv0,ib,ie,jb,je,kb,ke)
        call dump3('qc0',qc0,ib,ie,jb,je,kb,ke)
        call dump3('qi0',qi0,ib,ie,jb,je,kb,ke)
        call dump3('rr0',rr0,ib,ie,jb,je,kb,ke)
        call dump3('rf0',rf0,ib,ie,jb,je,kb,ke)
        call dump3('rrf0',rrf0,ib,ie,jb,je,kb,ke)
        call dump3('u0',u0,ib,ie+1,jb,je,kb,ke)
        call dump3('v0',v0,ib,ie,jb,je+1,kb,ke)
        call dump3('ua',ua,ib,ie+1,jb,je,kb,ke)
        call dump3('va',va,ib,ie,jb,je+1,kb,ke)
        call dump3('wa',wa,ib,ie,jb,je,kb,ke+1)
        call dump3('ppi',ppi,ib,ie,jb,je,kb,ke)
        call dump3('tha',tha,ib,ie,jb,je,kb,ke)
        call dump3('rho',rho,ib,ie,jb,je,kb,ke)
        call dump3('prs',prs,ib,ie,jb,je,kb,ke)
        if( imoist.eq.1 )then
          do n=1,numq
            call dump3('qa_'//trim(qname(n)),qa(ibm,jbm,kbm,n),ibm,iem,jbm,jem,kbm,kem)
          enddo
        endif
        call dump3('tkea',tkea,ibt,iet,jbt,jet,kbt,ket)
        call dump3('kmh',kmh,ibc,iec,jbc,jec,kbc,kec)
        call dump3('kmv',kmv,ibc,iec,jbc,jec,kbc,kec)
        call dump3('khh',khh,ibc,iec,jbc,jec,kbc,kec)
        call dump3('khv',khv,ibc,iec,jbc,jec,kbc,kec)
        call dump3('rain',rain,ib,ie,jb,je,1,nrain)
        if( sfcmodel.ge.1 )then
          ! surface layer (2D arrays, dumped with a third dimension 1:1):
          ! inputs carried across steps, then what sfc_and_turb computed
          ! from the previous state (as for kmh..khv)
          call dump3('tsk',tsk,ib,ie,jb,je,1,1)
          call dump3('znt',znt,ib,ie,jb,je,1,1)
          call dump3('xland',xland,ib,ie,jb,je,1,1)
          call dump3('mavail',mavail,ibl,iel,jbl,jel,1,1)
          call dump3('ust',ust,ib,ie,jb,je,1,1)
          call dump3('u1',u1,ib,ie,jb,je,1,1)
          call dump3('v1',v1,ib,ie,jb,je,1,1)
          call dump3('s1',s1,ib,ie,jb,je,1,1)
          call dump3('cd',cd,ib,ie,jb,je,1,1)
          call dump3('ch',ch,ib,ie,jb,je,1,1)
          call dump3('cq',cq,ib,ie,jb,je,1,1)
          call dump3('thflux',thflux,ib,ie,jb,je,1,1)
          call dump3('qvflux',qvflux,ib,ie,jb,je,1,1)
          call dump3('psfc',psfc,ib,ie,jb,je,1,1)
          call dump3('u10',u10,ibl,iel,jbl,jel,1,1)
          call dump3('v10',v10,ibl,iel,jbl,jel,1,1)
          call dump3('s10',s10,ibl,iel,jbl,jel,1,1)
          call dump3('hfx',hfx,ibl,iel,jbl,jel,1,1)
          call dump3('qfx',qfx,ibl,iel,jbl,jel,1,1)
          call dump3('qsfc',qsfc,ibl,iel,jbl,jel,1,1)
          call dump3('wspd',wspd,ibl,iel,jbl,jel,1,1)
        endif
        if( sfcmodel.eq.2 .or. sfcmodel.eq.3 )then
          ! WRF/MM5 surface layers: the Monin-Obukhov scale they carry
          ! across steps and the SLAB soil temperatures
          call dump3('mol',mol,ibl,iel,jbl,jel,1,1)
          call dump3('tslb',tslb,ibl,iel,jbl,jel,1,num_soil_layers)
        endif
        if( use_pbl .and. ( ipbl.eq.1 .or. ipbl.eq.2 ) )then
          ! YSU (ipbl=1) or the simple PBL scheme (ipbl=2): boundary-layer
          ! depth (YSU, or gethpbl2) and the PBL tendencies that
          ! sfc_and_turb computed from the previous state (as for kmh..khv)
          call dump3('hpbl',hpbl,ibl,iel,jbl,jel,1,1)
          call dump3('thpten',thpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qvpten',qvpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qcpten',qcpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('upten',upten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('vpten',vpten,ibb,ieb,jbb,jeb,kbb,keb)
        endif
        if( sfcmodel.eq.6 )then
          ! MYNN surface layer: the stability arrays it carries across steps
          call dump3('mol',mol,ibl,iel,jbl,jel,1,1)
          call dump3('rmol',rmol,ibl,iel,jbl,jel,1,1)
          call dump3('zol',zol,ibl,iel,jbl,jel,1,1)
          call dump3('ustm',ustm,ibl,iel,jbl,jel,1,1)
          ! and the SLAB soil temperatures (SLAB runs after it)
          call dump3('tslb',tslb,ibl,iel,jbl,jel,1,num_soil_layers)
        endif
        if( use_pbl .and. (ipbl.eq.4 .or. ipbl.eq.5) )then
          ! MYNN PBL (ipbl=4: level 2.5, ipbl=5: level 3): hpbl, the PBL
          ! tendencies and diffusivities that sfc_and_turb computed from the
          ! previous state (as for kmh..khv), the scheme's carried state, and
          ! qke3d (qke advected by solve2, the next MYNN call's input)
          call dump3('hpbl',hpbl,ibl,iel,jbl,jel,1,1)
          call dump3('thpten',thpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qvpten',qvpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qcpten',qcpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qipten',qipten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('upten',upten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('vpten',vpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('xkzh',xkzh,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('xkzm',xkzm,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qke',qke,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('qke3d',qke3d,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('tsq',tsq,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('qsq',qsq,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('cov',cov,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('sh3d',sh3d,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('sm3d',sm3d,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('el_pbl',el_pbl,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('qc_bl',qc_bl,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('qi_bl',qi_bl,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('cldfra_bl',cldfra_bl,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
          call dump3('dqke',dqke,ibmynn,iemynn,jbmynn,jemynn,kbmynn,kemynn)
        endif
        if( sfcmodel.eq.7 )then
          ! MYJ surface layer: the stability arrays, the SLAB soil
          ! temperatures, the roughness/viscous-sublayer state it carries
          ! across steps (thz0, qz0, uz0, vz0, akhs, akms, zntmyj; z0base is
          ! constant) and its 10-m diagnostics
          call dump3('mol',mol,ibl,iel,jbl,jel,1,1)
          call dump3('rmol',rmol,ibl,iel,jbl,jel,1,1)
          call dump3('tslb',tslb,ibl,iel,jbl,jel,1,num_soil_layers)
          call dump3('thz0',thz0,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('qz0',qz0,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('uz0',uz0,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('vz0',vz0,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('akhs',akhs,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('akms',akms,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('zntmyj',zntmyj,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('z0base',z0base,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('th10',th10,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('q10',q10,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('u10e',u10e,ibmyj,iemyj,jbmyj,jemyj,1,1)
          call dump3('v10e',v10e,ibmyj,iemyj,jbmyj,jemyj,1,1)
        endif
        if( use_pbl .and. ipbl.eq.6 )then
          ! MYJ PBL: hpbl, the PBL tendencies and diffusivities that
          ! sfc_and_turb computed from the previous state (as for kmh..khv),
          ! the TKE and mixing length it carries, and the mixed-layer depth
          call dump3('hpbl',hpbl,ibl,iel,jbl,jel,1,1)
          call dump3('thpten',thpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qvpten',qvpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qcpten',qcpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('qipten',qipten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('upten',upten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('vpten',vpten,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('xkzh',xkzh,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('xkzm',xkzm,ibb,ieb,jbb,jeb,kbb,keb)
          call dump3('tke_myj',tke_myj,ibmyj,iemyj,jbmyj,jemyj,kbmyj,kemyj)
          call dump3('el_myj',el_myj,ibmyj,iemyj,jbmyj,jemyj,kbmyj,kemyj)
          call dump3('mixht',mixht,ibmyj,iemyj,jbmyj,jemyj,1,1)
        endif
        call dump_end()
