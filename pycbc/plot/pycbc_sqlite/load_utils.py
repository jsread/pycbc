#! /usr/bin/env python

import sqlite3
import numpy

from pycbc import distributions
from pycbc.plot import data_utils

import lal
from glue import segments
from pylal import printutils

def get_r_distribution_from_inspinj(connection):
    """
    Gets the distance distribution that was given to inspinj from the
    process_params table.
    """
    sqlquery = """
        SELECT
            process_id, param, value
        FROM
            process_params
        WHERE
            program == "inspinj" AND
            (
                param == "--min-distance" OR
                param == "--max-distance" OR
                param == "--d-distr" OR
                param == "--dchirp-distr"
            )
        """
    rdistrs = {}
    for process_id, param, value in connection.cursor().execute(sqlquery):
        # order of storing things is type, distribution, min, max
        rdistrs.setdefault(process_id, ['', '', 0., 0.])
        if param == "--d-distr":
            rdistrs[process_id][0] = 'distance'
            rdistrs[process_id][1] = value
        if param == "--dchirp-distr":
            rdistrs[process_id][0] = "chirp_dist"
            rdistrs[process_id][1] = "uniform"
        if param == "--min-distance":
            rdistrs[process_id][2] = float(value)/1000. # convert kpc to Mpc
        if param == "--max-distance":
            rdistrs[process_id][3] = float(value)/1000.
    return rdistrs


def get_dist_weights_from_inspinj_distr(result, distr_type, distribution,
        r1, r2):
    """
    FIXME: the scale_fac and weights are copied from 
    randr_by_snr; these functions should be moved to library
    location instead
    """
    r = result.injection.distance
    if distr_type == "chirp_dist":
        # scale r1 and r2 by the chirp mass
        scale_fac = (result.injection.mchirp/\
            (2.8 * 0.25**0.6))**(5./6)
        r1 = r1*scale_fac
        r2 = r2*scale_fac
    if distribution == "volume":
        min_vol = (4./3)*numpy.pi*r1**3.
        vol_weight = (4./3)*numpy.pi*(r2**3. - r1**3.)
    elif distribution == "uniform":
        min_vol = (4./3)*numpy.pi*r1**3.
        vol_weight = 4.*numpy.pi*(r2-r1) * r**2. 
    elif distribution == "log10":
        min_vol = (4./3)*numpy.pi*r1**3.
        vol_weight = 4.*numpy.pi * r**3. * numpy.log(r2/r1)
    else:
        raise ValueError("unrecognized distribution %s" %(
            distribution))
    return min_vol, vol_weight


def cull_injection_results(results, primary_arg='false_alarm_rate',
        primary_rank_by='max', secondary_arg='new_snr',
        secondary_rank_by='min'):
    """
    Given a list of injection results in which the injections are mapped to
    multiple singles, picks the more significant one based on the primary_arg.
    If the two events have the same value in the primary arg, the secondary
    arg is used.
    """
    # get the correct operator to use
    if not (primary_rank_by == 'max' or primary_rank_by == 'min'):
        raise ValueError("unrecognized primary_rank_by %s; " %(
            primary_rank_by) + 'options are "max" or "min"')
    if not (secondary_rank_by == 'max' or secondary_rank_by == 'min'):
        raise ValueError("unrecognized secondary_rank_by %s; " %(
            secondary_rank_by) + 'options are "max" or "min"')
    # find the repeated entries
    sorted_results = sorted(results,
        key=lambda x: int(x.simulation_id.split(':')[-1]))
    id_map = {}
    duplicates = {}
    this_count = 1
    for ii,this_result in enumerate(sorted_results):
        if ii+1 < len(sorted_results) and \
                sorted_results[ii+1].simulation_id == \
                this_result.simulation_id:
            this_count += 1
        elif this_count > 1:
            # pick out the loudest out of the repeated values
            this_group = sorted_results[ii-(this_count-1):ii+1]
            primaries = numpy.array([data_utils.get_arg(x, primary_arg) \
                for x in this_group])
            if primary_rank_by == 'min':
                keep_idx = numpy.where(primaries == primaries.min())[0]
            else:
                keep_idx = numpy.where(primaries == primaries.max())[0]
            if len(keep_idx) > 1:
                secondaries = numpy.array([data_utils.get_arg(this_group[jj],
                    secondary_arg) for jj in keep_idx])
                # note: this will just keep the first event if the secondaries
                # are equal
                if secondary_rank_by == 'min':
                    secondary_idx = secondaries.argmin()
                else:
                    secondary_idx = secondaries.argmax()
                keep_idx = keep_idx[secondary_idx]
            else:
                keep_idx = keep_idx[0]
            # set this_result to the desired one; also set this_count = 0 for
            # the next group
            this_result = this_group[keep_idx]
            this_count = 1
        if this_count == 1:
            id_map[this_result.database, this_result.simulation_id] = \
                this_result
    return id_map.values(), id_map

    
# FIXME: add template info correctly
def get_injection_results(filenames, map_label,
        include_missed_injections=True, load_inj_distribution=True,
        load_vol_weights_from_inspinj=True,
        cull_primary_arg='false_alarm_rate', cull_primary_rank_by='min',
        cull_secondary_arg='new_snr', cull_secondary_rank_by='max',
        verbose=False):
    sqlquery = """
        SELECT
            sim.process_id, sim.waveform, sim.simulation_id,
            sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
            sim.spin2x, sim.spin2y, sim.spin2z,
            sim.distance, sim.inclination, sim.alpha1, sim.alpha2,
            tmplt.event_id, tmplt.mass1, tmplt.mass2,
            tmplt.spin1x, tmplt.spin1y, tmplt.spin1z,
            tmplt.spin2x, tmplt.spin2y, tmplt.spin2z,
            res.false_alarm_rate, res.combined_far, res.snr,
            res.ifos, res.coinc_event_id,
            -- experiment information
            experiment.instruments, exsumm.duration
        FROM
            coinc_inspiral AS res
        JOIN
            sngl_inspiral AS tmplt, coinc_event_map AS mapA
        ON
            mapA.coinc_event_id == res.coinc_event_id AND
            mapA.event_id == tmplt.event_id AND
            mapA.table_name == "sngl_inspiral"
            -- FIXME!
            AND tmplt.ifo == "H1"
        JOIN
            sim_inspiral AS sim, coinc_event_map AS mapB,
            coinc_event_map AS mapC
        ON
            sim.simulation_id == mapB.event_id AND
            mapB.coinc_event_id == mapC.coinc_event_id AND
            mapC.event_id == res.coinc_event_id
        JOIN
            coinc_definer AS cdef, coinc_event AS cev
        ON
            cdef.coinc_def_id == cev.coinc_def_id AND
            cev.coinc_event_id == mapB.coinc_event_id
        JOIN
            experiment, experiment_summary AS exsumm, experiment_map AS exmap
        ON
            exmap.coinc_event_id == res.coinc_event_id AND
            exmap.experiment_summ_id == exsumm.experiment_summ_id AND
            exsumm.experiment_id == experiment.experiment_id AND
            exsumm.datatype == "simulation" AND
            exsumm.sim_proc_id == sim.process_id
        WHERE
            cdef.description == ?
    """
    results = []
    idx = 0
    inj_dists = {}
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        cursor = connection.cursor()
        # if getting the distance distributions from inspinj, get it now
        if load_vol_weights_from_inspinj:
            rdistrs = get_r_distribution_from_inspinj(connection)
        for (sim_proc_id, apprx, sim_id, m1, m2, s1x, s1y, s1z,
                s2x, s2y, s2z, dist, inc, min_vol, vol_weight,
                tmplt_evid, tmplt_m1, tmplt_m2,
                tmplt_s1x, tmplt_s1y, tmplt_s1z,
                tmplt_s2x, tmplt_s2y, tmplt_s2z,
                uncombined_far, combined_far, snr, ifos, ceid,
                ifos_on, livetime) in \
                cursor.execute(sqlquery, (map_label,)):
            thisRes = data_utils.Result()
            # id information
            thisRes.unique_id = idx
            thisRes.database = thisfile 
            thisRes.event_id = ceid
            idx += 1
            # Set the injection parameters: we'll make the injection
            # the psuedo class so we can access its attributes directly
            thisRes.set_psuedoattr_class(thisRes.injection)
            thisRes.injection.simulation_id = sim_id
            thisRes.injection.approximant = apprx
            # ensure that m1 is always > m2
            if m2 > m1:
                thisRes.injection.mass1 = m2
                thisRes.injection.mass2 = m1
                thisRes.injection.spin1x = s2x
                thisRes.injection.spin1y = s2y
                thisRes.injection.spin1z = s2z
                thisRes.injection.spin2x = s1x
                thisRes.injection.spin2y = s1y
                thisRes.injection.spin2z = s1z
            else:
                thisRes.injection.mass1 = m1
                thisRes.injection.mass2 = m2
                thisRes.injection.spin1x = s1x
                thisRes.injection.spin1y = s1y
                thisRes.injection.spin1z = s1z
                thisRes.injection.spin2x = s2x
                thisRes.injection.spin2y = s2y
                thisRes.injection.spin2z = s2z
            thisRes.injection.distance = dist
            thisRes.injection.inclination = inc
            if load_vol_weights_from_inspinj:
                # get the distribution that was used by inspinj
                distr_type, distribution, r1, r2 = rdistrs[sim_proc_id]
                min_vol, vol_weight = get_dist_weights_from_inspinj_distr(
                    thisRes, distr_type, distribution, r1, r2)
            # note: if not loading weights from inspinj, the alpha1 and
            # alpha2 columns will be used
            thisRes.injection.min_vol = min_vol
            thisRes.injection.vol_weight = vol_weight
            # set the template parameters
            thisRes.template.tmplt_id = tmplt_evid
            thisRes.template.mass1 = tmplt_m1
            thisRes.template.mass2 = tmplt_m2
            thisRes.template.spin1x = tmplt_s1x
            thisRes.template.spin1y = tmplt_s1y
            thisRes.template.spin1z = tmplt_s1z
            thisRes.template.spin2x = tmplt_s2x
            thisRes.template.spin2y = tmplt_s2y
            thisRes.template.spin2z = tmplt_s2z
            # statistics
            thisRes.new_snr = snr
            thisRes.uncombined_far = uncombined_far
            thisRes.false_alarm_rate = combined_far
            # experiment
            thisRes.instruments_on = ifos_on
            # get the injection distribution information
            if load_inj_distribution:
                try:
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        sim_proc_id]
                except KeyError:
                    # need to load the distribution
                    inj_dists[thisfile, sim_proc_id] = \
                        distributions.get_inspinj_distribution(connection,
                        sim_proc_id)
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        sim_proc_id]
            results.append(thisRes)

        # add the outright missed injections
        # we'll get the missed injections using printmissed
        missed_injections = printutils.printmissed(connection,
            'sim_inspiral', 'coinc_inspiral', map_label, 'inspiral',
                limit=None, verbose=False)
        # convert from the output of printmissed to Result type
        for row in missed_injections:
            thisRes = data_utils.Result()
            # id information
            thisRes.unique_id = idx
            thisRes.database = thisfile 
            thisRes.event_id = None
            idx += 1
            # Set the injection parameters: we'll make the injection
            # the psuedo class so we can access its attributes directly
            thisRes.set_psuedoattr_class(thisRes.injection)
            thisRes.injection.simulation_id = row.simulation_id
            thisRes.injection.approximant = row.waveform
            # ensure that m1 is always > m2
            if row.mass2 > row.mass1:
                thisRes.injection.mass1 = row.mass2
                thisRes.injection.mass2 = row.mass1
                thisRes.injection.spin1x = row.spin2x
                thisRes.injection.spin1y = row.spin2y
                thisRes.injection.spin1z = row.spin2z
                thisRes.injection.spin2x = row.spin1x
                thisRes.injection.spin2y = row.spin1y
                thisRes.injection.spin2z = row.spin1z
            else:
                thisRes.injection.mass1 = row.mass1
                thisRes.injection.mass2 = row.mass2
                thisRes.injection.spin1x = row.spin1x
                thisRes.injection.spin1y = row.spin1y
                thisRes.injection.spin1z = row.spin1z
                thisRes.injection.spin2x = row.spin2x
                thisRes.injection.spin2y = row.spin2y
                thisRes.injection.spin2z = row.spin2z
            thisRes.injection.distance = row.distance
            thisRes.injection.inclination = row.inclination
            if load_vol_weights_from_inspinj:
                # get the distribution that was used by inspinj
                distr_type, distribution, r1, r2 = rdistrs[row.process_id]
                min_vol, vol_weight = get_dist_weights_from_inspinj_distr(
                    thisRes, distr_type, distribution, r1, r2)
            else:
                min_vol = row.alpha1
                weight = row.alpha2
            thisRes.injection.min_vol = min_vol
            thisRes.injection.vol_weight = vol_weight
            # since it wasn't found, the template is None
            thisRes.template = None
            # statistics: just set to bounds
            thisRes.new_snr = 0.
            thisRes.uncombined_far = thisRes.false_alarm_rate = numpy.inf
            # experiment
            thisRes.instruments_on = row.instruments_on
            # get the injection distribution information
            if load_inj_distribution:
                try:
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        row.process_id]
                except KeyError:
                    # need to load the distribution
                    inj_dists[thisfile, row.process_id] = \
                        distributions.get_inspinj_distribution(connection,
                        row.process_id)
                    thisRes.injection.mass_distr = inj_dists[thisfile,
                        row.process_id]
            results.append(thisRes)
        connection.close()

    if verbose:
        print >> sys.stdout, ""
    
    # cull the results for duplicated entries and create an id_map
    results, id_map = cull_injection_results(results,
        primary_arg=cull_primary_arg,
        primary_rank_by=cull_primary_rank_by,
        secondary_arg=cull_secondary_arg,
        secondary_rank_by=cull_secondary_rank_by)
    # standdrd id map is to point to the idx, not the result
    id_map = dict([[x.simulation_id, ii] for ii,x in enumerate(results)])
    return results, id_map


def get_livetime(filenames):
    """
    Gets the total live time from a list of sqlite databases produced by the
    pycbc workflow. Live times are only added if the experiments in multiple
    filenames do not have overlapping gps end times.
    """
    sqlquery = """
        SELECT
            exp.experiment_id, exp.instruments,
            exp.gps_start_time, exp.gps_end_time,
            exsumm.veto_def_name, exsumm.datatype, exsumm.duration
        FROM
            experiment as exp
        JOIN
            experiment_summary as exsumm
        ON
            exp.experiment_id == exsumm.experiment_id
        """
    livetimes = {}
    for filename in filenames:
        connection = sqlite3.connect(filename)
        thisdict = {}
        for eid, instruments, gps_start, gps_end, vetoes, datatype, duration \
                in connection.cursor().execute(sqlquery):
            exkey = (eid, instruments, vetoes, datatype)
            this_seg = segments.segment(gps_start, gps_end)
            thisdict.setdefault(exkey, [this_seg, None])
            # for the first one, always just add the duration
            if thisdict[exkey][1] is None:
                thisdict[exkey][1] = duration 
            # if datatype is slide add the times
            elif datatype == "slide":
                thisdict[exkey][1] += duration
            # otherwise, check that the livetime is the same 
            elif duration != thisdict[exkey][1]:
                raise ValueError("unequal durations for " + exkey + \
                    "in file %s" %(filename))
        connection.close()
        # add to the master list of livetimes
        for ((_, instruments, vetoes, datatype), [this_seg, dur]) in \
                thisdict.items():
            # we'll make a dict of dict with datatype being the primary key
            livetimes.setdefault(datatype, {})
            exkey = (instruments, vetoes)
            try:
                seg_list, _ = livetimes[datatype][exkey]
            except KeyError:
                # doesn't exist, create
                seg_list = segments.segmentlist([])
                livetimes[datatype][exkey] = [seg_list, 0]
                
            # check that this_seg does not intersect with the segments
            # from the other files
            if seg_list.intersects_segment(this_seg):
                raise ValueError("Experiment (%s, %s, %s) " % exkey + \
                    "has overlapping gps times in multiple files")

            # add the duration and update the segment list
            seg_list.append(this_seg)
            seg_list.coalesce()
            livetimes[datatype][exkey][1] += dur

    return livetimes
