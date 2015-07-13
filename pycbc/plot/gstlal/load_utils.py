#! /usr/bin/env python

import sqlite3
import numpy
import itertools

from pycbc import distributions
from pycbc.plot import data_utils
from pycbc.plot.pycbc_sqlite.load_utils import \
    get_r_distribution_from_inspinj, get_dist_weights_from_inspinj_distr
from pycbc.plot.data_utils import cull_injection_results

import lal
from glue import segments
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw import table
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments


def get_all_ifo_combinations(ifos):
    """
    Given a list of ifos, returns all possible combinations.
    """
    return [combo for n in range(2, len(ifos)+1) \
        for combo in itertools.combinations(sorted(ifos), n)]


def time_within_segments(end_time, seglist=None):
    """
    Determines if a particular time was within the given segment list. If no
    segment list is provided, just returns true.

    Parameters
    ----------
    end_time: {float|lal.LIGOTimeGPS}
        GPS seconds of the end time.
    seglist: {None|glue.segments_list}
        Segment list to check.

    Returns
    -------
    bool
        Whether or not the time was in the segments.
    """
    return segments is None or end_time in seglist


def get_injections_as_sim_inspiral(connection):
    """
    Gets all of the injections in a database from the sim_inspiral table.

    Parameters
    ----------
    connection: sqlite3.connection
        Connection to a sqlite database.
    
    Returns
    -------
    lsctables.SimInspiralTable
        All of the injections as a sim_inspiral table.
    """
    # get the mapping of a record returned by the database to a sim
    # inspiral row. Note that this is DB dependent potentially, so always
    # do this!
    xmldoc = dbtables.get_xml(connection)
    make_sim_inspiral = lsctables.table.get_table(xmldoc,
        lsctables.SimInspiralTable.tableName).row_from_cols
    sqlquery = """
        SELECT
            *
        FROM
            sim_inspiral
        """
    injections = map(make_sim_inspiral,
        connection.cursor().execute(sqlquery).fetchall())
    xmldoc.unlink()
    return injections 


def get_analysis_segments(xmldoc):
    """
    Gets the analysis segments from an output database from a gstlal run.
    """
    # get the data segments
    analysis_segs = ligolw_segments.segmenttable_get_by_name(xmldoc,
        "datasegments").coalesce()
    # intersect with the trigger segments (?)
    analysis_segs &= ligolw_segments.segmenttable_get_by_name(xmldoc,
        "triggersegments").coalesce()
    # subtract out the vetoes
    analysis_segs -= ligolw_segments.segmenttable_get_by_name(xmldoc,
        "vetoes").coalesce()
    return analysis_segs


def injection_was_made(inj, analysis_segments, check_ifos):
    """
    Given an injection (in sim_inspiral form) and a segmentlistdict of analysis
    segments, checks if the injection occured in the segments by checking
    the end time in each of the ifos in check_ifos.
    """
    return all([time_within_segments(
            inj.get_end(ifo[0]), analysis_segments[ifo])
        for ifo in check_ifos])


def get_made_injections(injections, analysis_segments, instrument_combos=None):
    """
    Given a list of injections (in sim_inspiral table form) and a
    segmentlistdict of analysis segments, returns the injections that occured
    in the segments.
    """
    # if instrument_combos is not specified, use all of the ifos that are in
    # the analysis segmentlist dict
    if instrument_combos is None:
        instrument_combos = get_all_ifo_combinations(analysis_segments.keys())
    made_injections = {}
    # a list of indices to keep track of whether or not an injection was found
    # in another instrument time
    already_made = numpy.zeros(len(injections), dtype=bool)
    indices = numpy.arange(len(injections), dtype=int)
    for this_combo in instrument_combos:
        is_made = numpy.array([
            injection_was_made(injections[ii], analysis_segments, this_combo) \
            for ii in indices if not already_made[ii]])
        # update the already_found list of indices
        already_made += is_made
        # add to the list of made injections
        made_injections[this_combo] = [injections[ii] \
            for ii in numpy.where(is_made)[0]]
    return made_injections


def get_found_injection_info(connection, coinc_table='coinc_inspiral',
        map_label="sim_inspiral<-->coinc_event coincidences (nearby)"):
    """
    Gets simulation_ids of found injections, along with all of the information
    about the injection from the given coinc_table.
    """
    sqlquery = """
        SELECT
            sim.simulation_id, res.*
        FROM
            %s AS res""" %(coinc_table) + """
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
        WHERE
            cdef.description == ?
        """
    # way to create the coinc row from the database outputs
    xmldoc = dbtables.get_xml(connection)
    make_coinc_row = lsctables.table.get_table(xmldoc,
        coinc_table).row_from_cols
    found_info = {}
    for values in connection.cursor().execute(
            sqlquery, (map_label,)).fetchall():
        sim_id = values[0]
        coinc_info = make_coinc_row(values[1:])
        try:
            found_info[sim_id].append(coinc_info)
        except KeyError:
            found_info[sim_id] = [coinc_info]
    xmldoc.unlink()
    return found_info


def get_most_significant_result(found_coincs,
        primary_arg='false_alarm_rate', primary_rank_by='min',
        secondary_arg='ranking_stat', secondary_rank_by='max'):
    primaries = numpy.array([data_utils.get_arg(x, primary_arg) \
        for x in found_coincs])
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
    return found_coincs[keep_idx]


def get_injection_results(filenames,
        map_label="sim_inspiral<-->coinc_event coincidences (nearby)",
        include_missed_injections=True, load_inj_distribution=True,
        load_vol_weights_from_inspinj=True,
        cull_primary_arg='false_alarm_rate', cull_primary_rank_by='min',
        cull_secondary_arg='ranking_stat', cull_secondary_rank_by='max',
        verbose=False):
    results = []
    idx = 0
    inj_dists = {}
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        connection = sqlite3.connect(thisfile)
        xmldoc = dbtables.get_xml(connection)
        # get the analysis segments
        analysis_segs = get_analysis_segments(xmldoc)
        # get all of the made injections
        # This will return the on instruments along with the injection list
        made_injections = get_made_injections(
            get_injections_as_sim_inspiral(connection),
            analysis_segs)
        # get the found injection information
        found_injections = get_found_injection_info(connection,
            coinc_table='coinc_inspiral', map_label=map_label)
        # if getting the distance distributions from inspinj, get it now
        if load_vol_weights_from_inspinj:
            rdistrs = get_r_distribution_from_inspinj(connection)
        # convert to Result class
        for on_instruments, injections in made_injections.items():
            for row in injections:
                thisRes = data_utils.Result()
                # id information
                thisRes.unique_id = idx
                thisRes.database = thisfile 
                thisRes.event_id = None
                sim_proc_id = str(row.process_id)
                simid = str(row.simulation_id)
                idx += 1
                # Set the injection parameters: we'll make the injection
                # the psuedo class so we can access its attributes directly
                thisRes.set_psuedoattr_class(thisRes.injection)
                thisRes.injection.simulation_id = simid
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
                    distr_type, distribution, r1, r2 = rdistrs[sim_proc_id]
                    min_vol, vol_weight = get_dist_weights_from_inspinj_distr(
                        thisRes, distr_type, distribution, r1, r2)
                else:
                    min_vol = row.alpha1
                    weight = row.alpha2
                thisRes.injection.min_vol = min_vol
                thisRes.injection.vol_weight = vol_weight
                # FIXME: just setting the template to None for all injections
                # for now
                thisRes.template = None
                # if found, set the relevant statistics
                try:
                    found_coincs = found_injections[simid]
                    # pull out the most significant one
                    found_coinc = get_most_significant_result(found_coincs,
                        primary_arg=cull_primary_arg,
                        primary_rank_by=cull_primary_rank_by,
                        secondary_arg=cull_secondary_arg,
                        secondary_rank_by=cull_secondary_rank_by)
                    thisRes.ranking_stat = found_coinc.snr
                    # gstlal stores its FARs in Hz, so convert to years
                    thisRes.false_alarm_rate = found_coinc.combined_far * \
                        lal.YRJUL_SI
                except KeyError:
                    # injection was missed, just set to bounds
                    thisRes.ranking_stat = 0.
                    thisRes.false_alarm_rate = numpy.inf
                # experiment
                thisRes.instruments_on = on_instruments
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
        xmldoc.unlink()
        connection.close()

    if verbose:
        print >> sys.stdout, ""
    
    # standdrd id map is to point to the idx, not the result
    id_map = dict([[x.simulation_id, ii] for ii,x in enumerate(results)])
    return results, id_map


def combine_analysis_segments(all_segs, err_on_partial_overlap=True):
    """
    Given analysis segments from multiple files, combines them into a single
    coalesced segment list. 

    Parameters
    ----------
    all_segs: dict
        A dictionary of the analysis segments from each file. Keys should be
        the filenames, values a segments.segmentlist of the analysis segments
        in that file.
    err_on_partial_overlap: {True|bool}
        If True, will raise an error if any two of the files have partially
        overlapping analysis segments.

    Returns
    -------
    masterlist: segments.segmentlist
        Coalesced segments of all the files combined.
    """
    # combine the segments from all of the files: if two files have the same
    # analysis segments, we only use one. If two files have only partially
    # intersecting segments, we raise an error if desired. Otherwise, we add
    # the segments.
    masterlist = segments.segmentlist([])
    for filename,seg_list in all_segs.items():
        if seg_list != masterlist:
            if err_on_partial_overlap and masterlist.intersects(seg_list):
                raise ValueError("file %s partially overlaps the other files"\
                    %(filename))
            masterlist.extend(seg_list)
            masterlist.coalesce()
    return masterlist


def get_all_analysis_segments(filenames, err_on_partial_overlap=True):
    """
    Given a list of filenames, gets a coaleseced segment list dict of the
    combined analysis segments, keyed by the on instruments.
    
    Parameters
    ----------
    filenames: list
        List of filenames to retrieve the segments from.

    err_on_partial_overlap: {True|bool}
        If True, will raise an error if any two of the files have partially
        overlapping analysis segments.

    Returns
    -------
    all_analysis_segments: segments.segmentlistdict
        Segmentlist dict of coalesced segments of all the files combined,
        keyed by all possible on_instrument combinations in the lists.
    """
    all_segs = {}
    for filename in filenames:
        connection = sqlite3.connect(filename)
        xmldoc = dbtables.get_xml(connection)
        analysis_segs = get_analysis_segments(xmldoc)
        # construct segment lists for all possible ifo combos
        for ifocombo in get_all_ifo_combinations(analysis_segs.keys()):
            all_segs.setdefault(ifocombo, {})
            all_segs[ifocombo][filename] = analysis_segs.intersection(
                ifocombo)
        xmldoc.unlink()
        connection.close()
    return segments.segmentlistdict([
        [ifocombo, combine_analysis_segments(all_files,
            err_on_partial_overlap=err_on_partial_overlap)] \
        for ifocombo, all_files in all_segs.items()])


def get_livetime(filenames):
    """
    Gets the total live time from a list of sqlite databases produced by the
    workflow. Live times are only added if the experiments in multiple
    filenames do not have overlapping gps end times. If any two of the files
    have paritally overlapping analysis segments, an error is raised.

    Parameters
    ----------
    filenames: list
        List of filenames to retrieve the livetime from.

    Returns
    -------
    livetime: dict
        Dictionary of the livetimes, keyed by all possible on_instrument
        combinations across the files.
    """
    return dict([ [ifocombo, float(abs(seglist))] \
        for ifocombo,seglist in get_all_analysis_segments(filenames).items()])
