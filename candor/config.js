// NeXus layout for the instrument
// Group names are defined by dictionaries:
//   name$NXgroup: { fields } 
// Fields are defined by 
//   ->path  is a link to a DAS log entry
//   $Field  means substitute the configuration value
//   name$NXgroup: {}  defines a nexus group
//   
// The following fields are automatic:
//   start_time: timestamp on first State record
//   end_time: timestamp on final Counts record
//   duration: end_time - start_time
//   collection_time: sum(count_time)
//   sample.measurement: ->controller.sensor
//   sample.measurement_log: ->controller.sensor_log
//
// The sample environment fields are generated from the sensor 
// definitions, with the field name in sample coming from
// the {measurement: 'fieldname'} in the configuration.
//

var entry = {
definition: "NXrefscan",
facility: "NCNR",
title: "->trajectory.name",
experiment_description: "->experiment.title",
experiment_identifier: "->experiment.proposalId",
// run_cycle: "->experiment.runCycle",
program_data$NXnote: { // [NCNR] additional program data
    type: "application/json",
    data: "->trajectory.config",
    file_name: "->?trajectory.configFile",
    description: "Additional program data, such as the script file which the program ran",
    },
sample$NXsample: {
    name: "->sample.name",
    description: "->sample.description",
    polar_angle: "->detectorAngle.softPosition",
    rotation_angle: "->sampleAngle.softPosition",
    Qx: "->q.x",
    Qz: "->q.z",

    // Environment variables per point link to the environment log average
    temperature: "->?sample/temperature_log/average_value",
    magnetic_field: "->?sample/magenetic_field_log/average_value",
    pressure: "->?sample/pressure_log/average_value",

    // DAS_logs contain the NXlog entry for 
    temperature_log: "->?temperature",
    pressure_log: "->?pressureChamber",
    },
control$NXmonitor: {
    count_start: "->counter.startTime",
    count_end: "->counter.stopTime",
    count_time: "->counter.liveTime",
    monitor_preset: "->counter.monitorPreset",
    count_time_preset: "->counter.timePreset",
    detector_preset: "->counter.roiPreset",
    detector_counts: "->counter.liveROI",
    monitor_counts: "->counter.liveMonitor",
    // Monitor properties
    efficiency: "0.1 %",
    absorption: "0 %",
    sampled_fraction: {value:0.1, units:""},
    type: "??"  // Type of monitor        
    },
instrument$NXinstrument: {
    name: "NCNR Magik",
    source$NXsource: {
        distance: {value:-694, units:"cm"},
        name: "NCNR",
        type: "Reactor Neutron Source",
        probe: "neutron",
        power: "20 MW"
        },
    shutter$NXaperture: {
        description: "Beam shutter",
        width: "6.4 cm",
        height: "16 cm"
        },
    premonochromator_filter$NXfilter: {
        description: "Be",
        // nexus wants "in|out" but we store ICE "IN|OUT"
        // analysis prefers True|False
        // status: "->BeFilterControl.enumValue", // "in|out".
        // nexus wants 3x3 orientation matrix, but we only have tilt
        // and rotation columns; we will use polar angle (vertical
        // rotation) and azimuthal angle (horizontal rotation).
        // to beam direction) and azimuthal angle (rotation
        //distance: "?? cm", // documentation
        // If we were to add a temperature controlled filter, it would
        // need a sensor to record the value
        //temperature: "->filterTemperature.sensor0",
        //temperature_log: "->filterTemperature.sensor0_log"
        },
        
    monochromator$NXcrystal: {
        description: "Double bounce monochromator.",
        distance: "-5216.5 mm",
        //material: "->ei.material",
        //dspacing: "->ei.dSpacing",
        wavelength: "4.75 angstom",
        // wavelength_error is 1-sigma:
        wavelength_error: "0.03 angstrom"
        },
    //
    // Beam monitor goes here
    //
    presample_polarizer$NXpolarizer: {
        type: "He[3]"
        },
//    presample_flipper$NXflipper: {
//        type: "current sheet",
//        comp_current: "->frontPol.cancelCurrent",
//        flip_current: "->frontPol.flipperCurrent",
//        guide_current: "->frontPol.guideCurrent"
//        },
    presample_slit1$NXaperture: {
        width: "->slitAperture1.softPosition",
        height: "->vertSlitAperture1.softPosition",
        distance: "-1759.0 mm"
        },
    presample_slit2$NXaperture: {
        width: "->slitAperture2.softPosition",
        distance: "-330 mm",
        //height: "->vertSlitAperture2.softPosition"
        },
    //
    // Sample goes here
    //
    predetector_polarizer$NXpolarizer: {
        type: "He[3]"
        },
//    predetector_flipper$NXflipper: {
//        type: "current sheet",
//        comp_current: "->backPol.cancelCurrent",
//        flip_current: "->backPol.flipperCurrent",
//        guide_current: "->backPol.guideCurrent"
//        },
    predetector_slit1$NXaperture: {
        //height: "->vertSlitAperture3.softPosition",
        width: "->slitAperture3.softPosition",
        //distance: "330 mm",
        },
    predetector_slit2$NXaperture: {
        //height: "->vertSlitAperture4.softPosition",
        width: "->slitAperture4.softPosition",
        //distance: "750 mm",
        },
    single_detector$NXdetector: {
        description: "Single tube detector used for high resolution, low background measurements.",
        type: "He[3]",
        //status: "->singleDetector.active", // "in|out"
        polar_angle: "->detectorAngle.softPosition",
        data: "->?pointDetector.counts",
        dead_time: {value: 8.737e-6, units: "s"},
        dead_time_error: {value: 0.034e-6, units: "s"},
        x_pixel_offset: {value: 0, units: "cm"},
        x_pixel_size: {value: 2.5, units: "cm"},
        y_pixel_offset: {value:0, units:"cm"},
        y_pixel_size: {value:15, units:"cm"}
        },
    PSD$NXdetector: {
        description: ".",
        //status: "->counter.active", // "in|out"
        polar_angle: "->detectorAngle.softPosition",
        data: "->?areaDetector.counts",
        // 512 x 608 values, with calibration for effective pixel size
        //x_pixel_offset: {value:0, units:"cm"},
        //x_pixel_size: {value:16.5, units:"cm"}
        //y_pixel_offset: {value:0, units:"cm"},
        //y_pixel_size: {value:16.5, units:"cm"}
        }
    }
}
