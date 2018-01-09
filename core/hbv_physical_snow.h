///    Copyright 2012 Statkraft Energi A/S
///
///    This file is part of Shyft.
///
///    Shyft is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///    Shyft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE.  See the    GNU Lesser General Public License for more details.
///
///    You should have received a copy of the GNU Lesser General Public License along with
/// Shyft, usually located under the Shyft root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.    If not, see <http://www.gnu.org/licenses/>.
///
/// Adapted from early enki method programmed by Kolbjørn Engeland and Sjur Kolberg
///


#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>

#include "time_series.h"
#include "utctime_utilities.h"

namespace shyft {
    namespace core {
        namespace hbv_physical_snow {
            using namespace std;

            static const double tol = 1.0e-10;

            /** \brief integrate function f given as linear interpolated
             * between the f_i(x_i) from a to b for a, b in x.  If
             * f_rhs_is_zero is set, f(b) = 0, unless b = x[i] for some i in
             * [0,n).
             */
            inline double integrate(vector<double> f, vector<double> x,
                                    size_t n, double a, double b,
                                    bool f_b_is_zero=false) {
                size_t left = 0;
                double area = 0.0;
                double f_l = 0.0;
                double x_l = a;
                while (a > x[left]) ++left;

                // Linear interpolation of start point
                if (fabs(a - x[left]) > 1.0e-8 && left > 0) {
                    --left;
                    f_l = ((f[left + 1] - f[left]) /
                           (x[left + 1] - x[left]) * (a - x[left]) + f[left]);
                } else
                    f_l = f[left];

                while (left < n - 1) {
                    if (b >= x[left + 1]) {
                        area += 0.5*(f_l + f[left + 1])*(x[left + 1] - x_l);
                        x_l = x[left + 1];
                        f_l = f[left + 1];
                        ++left;
                    }
                    else {
                        if (! f_b_is_zero)
                            area += (f_l + 0.5 * (f[left + 1] - f_l) /
                                     (x[left + 1] - x_l) * (b - x_l))*(b - x_l);
                        else
                            area += 0.5*f_l*(b - x_l);
                        break;
                    }
                }
                return area;
            }

            struct parameter {

                vector<double> s; // snow redistribution vector
                vector<double> intervals; // snow quantiles list 0, 0.25, 0.5, 1.0
                double tx = 0.0;
                double lw = 0.1;
                double cfr = 0.5;
                double wind_scale = 2.0;
                double wind_const = 1.0;
                double surface_magnitude = 30.0;
                double max_albedo = 0.9;
                double min_albedo = 0.6;
                double fast_albedo_decay_rate = 5.0;
                double slow_albedo_decay_rate = 5.0;
                double snowfall_reset_depth = 5.0;
                bool calculate_iso_pot_energy = false;


                void set_std_distribution_and_quantiles() {
                    double si[] = {1.0, 1.0, 1.0, 1.0, 1.0};
                    double ii[] = {0, 0.25, 0.5, 0.75, 1.0};
                    s.clear();s.reserve(5);
                    intervals.clear();intervals.reserve(5);
                    for(size_t i=0; i<5; ++i) {
                        s.push_back(si[i]);
                        intervals.push_back(ii[i]);
                    }
                }

                parameter(double tx=0.0, double lw=0.1, double cfr=0.5,
                        double wind_scale=2.0, double wind_const=1.0,
                        double surface_magnitude=30.0,
                        double max_albedo=0.9, double min_albedo=0.6,
                        double fast_albedo_decay_rate=5.0,
                        double slow_albedo_decay_rate=5.0,
                        double snowfall_reset_depth=5.0,
                        bool calculate_iso_pot_energy=false)
                            : tx(tx), lw(lw), cfr(cfr), wind_scale(wind_scale),
                              wind_const(wind_const),
                              surface_magnitude(surface_magnitude),
                              max_albedo(max_albedo),
                              min_albedo(min_albedo),
                              fast_albedo_decay_rate(fast_albedo_decay_rate),
                              slow_albedo_decay_rate(slow_albedo_decay_rate),
                              snowfall_reset_depth(snowfall_reset_depth),
                              calculate_iso_pot_energy(calculate_iso_pot_energy)
                {
                    set_std_distribution_and_quantiles();
                }

                parameter(const vector<double>& s,
                        const vector<double>& intervals,
                        double tx=0.0, double lw=0.1, double cfr=0.5,
                        double wind_scale=2.0, double wind_const=1.0,
                        double surface_magnitude=30.0,
                        double max_albedo=0.9, double min_albedo=0.6,
                        double fast_albedo_decay_rate=5.0,
                        double slow_albedo_decay_rate=5.0,
                        double snowfall_reset_depth=5.0,
                        bool calculate_iso_pot_energy=false)
                            : s(s), intervals(intervals), tx(tx), lw(lw),
                              cfr(cfr), wind_scale(wind_scale),
                              wind_const(wind_const),
                              surface_magnitude(surface_magnitude),
                              max_albedo(max_albedo),
                              min_albedo(min_albedo),
                              fast_albedo_decay_rate(fast_albedo_decay_rate),
                              slow_albedo_decay_rate(slow_albedo_decay_rate),
                              snowfall_reset_depth(snowfall_reset_depth),
                              calculate_iso_pot_energy(calculate_iso_pot_energy)
                {}

                void set_snow_redistribution_factors(
                        const vector<double>& values) {
                    s = values;
                }

                void set_snow_quantiles(const vector<double>& values) {

                    intervals = values;
                }
            };


            struct state {

                vector<double> albedo;
                vector<double> iso_pot_energy;

                double surface_heat = 30000.0;
                double swe = 0.0;
                double sca = 0.0;

                state() = default;

                state(const vector<double>& albedo,
                      const vector<double>& iso_pot_energy,
                      double surface_heat=30000.0, double swe=0.0,
                      double sca=0.0)
                    : albedo(albedo), iso_pot_energy(iso_pot_energy),
                      surface_heat(surface_heat), swe(swe), sca(sca)
                {}

                bool operator==(const state &x) const {
                    const double eps = 1e-6;
                    if (albedo.size() != x.albedo.size()) return false;
                    if (iso_pot_energy.size() != x.iso_pot_energy.size()) {
                        return false;
                    }
                    for (size_t i=0; i<albedo.size(); ++i) {
                        if (fabs(albedo[i] - x.albedo[i]) >= eps ||
                                fabs(iso_pot_energy[i] - x.iso_pot_energy[i]) >=
                                eps) {
                            return false;
                        }
                    }

                    return fabs(surface_heat - x.surface_heat) < eps
                        && fabs(swe - x.swe) < eps
                        && fabs(sca - x.sca) < eps;
                }
                x_serialize_decl();
            };

            struct response {
                double sca = 0.0;
                double storage = 0.0;
                double outflow = 0.0;
                state hps_state;
            };


            /** \brief Generalized quantile based HBV Snow model method, using
             * the physical melt model from Gamma-snow (refreeze is treated by multiplying potential_melt by a refreeze coefficient)
             *
             * This algorithm uses arbitrary quartiles to model snow. No checks are performed to assert valid input. The starting points of the
             * quantiles have to partition the unity, include the end points 0 and 1 and must be given in ascending order.
             *
             * \tparam P Parameter type, implementing the interface:
             *    - P.s() const --> vector<double>, snowfall redistribution vector
             *    - P.intervals() const --> vector<double>, starting points for the quantiles
             *    - P.lw() const --> double, max liquid water content of the snow
             *    - P.tx() const --> double, threshold temperature determining if precipitation is rain or snow
             *    - P.cx() const --> double, temperature index, i.e., melt = cx(t - ts) in mm per degree C
             *    - P.ts() const --> double, threshold temperature for melt onset
             *    - P.cfr() const --> double, refreeze coefficient, refreeze = cfr*cx*(ts - t)
             * \tparam S State type, implementing the interface:
             *    - S.albedo --> vector<double>, broadband snow reflectivity fraction in each snow bin.
             *    - S.iso_pot_energy --> vector<double>, accumulated energy assuming isothermal snow surface [J/m2]
             *    - S.surface_heat --> double, snow surface cold content [J/m2]
             *    - S.swe --> double, snow water equivalent of the snowpack [mm]
             *    - S.sca --> double, fraction of area covered by the snowpack [0,1]
             * \tparam R Response type, containing the member variables:
             *    - R.outflow --> double, the value of the outflow [mm]
             *    - R.storage --> double, the value of the swe [mm]
             *    - R.sca --> double, the value of the snow covered area
             *    - R.state --> shyft::core::hbv_physical_snow::state, containing
             *          the current state.
             */
            template<class P, class S, class R>
            class calculator {

              private:
                vector<double> sd;
                vector<double> I;
                vector<double> sp;
                vector<double> sw;
                size_t I_n;
                const double melt_heat = 333660.0;
                const double water_heat = 4180.0;
                const double ice_heat = 2050.0;
                const double sigma = 5.670373e-8;
                const double BB0{0.98*sigma*pow(273.15,4)};

                void refreeze(double &sp, double &sw, const double rain,
                        const double potmelt, const double lw) {
                    // Note that the above calculations might violate the mass
                    // balance due to rounding errors. A fix might be to
                    // replace sw by a sw_fraction, sp with s_tot, and compute
                    // sw and sp based on these.
                    if (sp > 0.0) {
                        if (sw + rain > -potmelt) {
                            sp -= potmelt;
                            sw += potmelt + rain;
                            if (sw > sp*lw) sw = sp*lw;
                        } else {
                            sp += sw + rain;
                            sw = 0.0;
                        }
                    }
                }


                void update_state(double &sp, double &sw, const double rain,
                        const double potmelt, const double lw) {
                    if (sp > potmelt) {
                        sw += potmelt + rain;
                        sp -= potmelt;
                        sw = std::min(sw, sp*lw);
                    } else if (sp > 0.0)
                        sp = sw = 0.0;
                }


                size_t sca_index(double sca) const {
                    for (size_t i = 0;  i < I_n - 1; ++i)
                        if (sca >= I[i] && sca < I[i + 1])
                            return i;
                    return I_n - 1;
                }


              public:
                calculator(const P& p, S& state) {
                    auto s = p.s;
                    I = p.intervals;
                    I_n = I.size();

                    const double mean = hbv_physical_snow::integrate(
                            s, I, I_n, I[0], I[I_n-1]);

                    sd = s;
                    for (auto &sd_ : sd) sd_ /= mean;

                    double swe = state.swe;
                    double sca = state.sca;

                    sp = vector<double>(I_n, 0.0);
                    sw = vector<double>(I_n, 0.0);
                    if (swe <= 1.0e-3 || sca <= 1.0e-3) {
                        state.swe = state.sca = 0.0;
                    } else {
                        for (size_t i=0; i<I_n; ++i)
                            sp[i] = sca < I[i] ? 0.0 : s[i] * swe;

                        swe = integrate(sp, I, I_n, 0.0, sca, true);

                        if (swe < state.swe) {
                            const double corr1 = state.swe/swe*p.lw;
                            const double corr2 = state.swe/swe*(1.0 - p.lw);
                            for (size_t i=0; i<I_n; ++i) {
                                sw[i] = corr1 * sp[i];
                                sp[i] *= corr2;
                            }
                        } else
                            sw = vector<double>(I_n, 0.0);
                    }
                }

                  /*
                  * \brief step the snow model forward from time t to t+dt, state, parameters and input
                  * updates the state and response upon return.
                  * \param s state of type S,in/out, ref template parameters
                  * \param r result of type R, output only, ref. template parameters
                  * \param t start time
                  * \param dt timespan
                  * \param p parameter of type P, ref template parameters
                  * \param T temperature degC, considered constant over timestep dt
                  * \param rad radiation
                  * \param prec precipitation in mm/h
                  * \param wind_speed in m/s
                  * \param rel_hum 0..1
                  */
                void step(S& s, R& r, shyft::time_series::utctime t,
                        shyft::time_series::utctimespan dt, const P& p,
                        const double T, const double rad,
                        const double prec_mm_h, const double wind_speed,
                        const double rel_hum) {

                    vector<double> iso_pot_energy = s.iso_pot_energy;
                    double sca = s.sca;
                    double swe = s.swe;
                    const double prec = prec_mm_h*dt/calendar::HOUR;
                    const double total_water = prec + swe;

                    double snow;
                    double rain;
                    if( T < p.tx ) {snow = prec; rain = 0.0;}
                    else           {snow = 0.0;rain = prec;}
                    if (fabs(snow + rain - prec) > 1.0e-8)
                        throw std::runtime_error("Mass balance violation!!!!");

                    swe += snow + sca * rain;
                    //Roughly the same as the 'early autumn scenario' of
                    //gamma_snow - i.e. no stored or precipitated snow
                    if (swe < hbv_physical_snow::tol) {
                        // Reset everything
                        r.outflow = total_water;
                        fill(begin(sp), end(sp), 0.0);
                        fill(begin(sw), end(sw), 0.0);
                        s.swe = 0.0;
                        s.sca = 0.0;

                        r.sca = 0.0;
                        r.storage = 0.0;

                        std::fill(s.albedo.begin(), s.albedo.end(),
                                  p.max_albedo);
                        s.surface_heat = 0.0;
                        std::fill(s.iso_pot_energy.begin(),
                                  s.iso_pot_energy.end(), 0.0);
                        return;
                    }

                    // Trivial case is out of the way, now more complicated

                    // State vars
                    vector<double> albedo = s.albedo;
                    double surface_heat = s.surface_heat;

                    // Response vars;
                    //double outflow = 0.0;

                    // Local variables

                    const double min_albedo = p.min_albedo;
                    const double max_albedo = p.max_albedo;
                    const double albedo_range = max_albedo - min_albedo;
                    const double dt_in_days = dt/double(calendar::DAY);
                    const double slow_albedo_decay_rate = (0.5 * albedo_range *
                            dt_in_days/p.slow_albedo_decay_rate);
                    const double fast_albedo_decay_rate = pow(2.0,
                            -dt_in_days/p.fast_albedo_decay_rate);

                    const double T_k = T + 273.15; // Temperature in Kelvin
                    const double turb = p.wind_scale*wind_speed + p.wind_const;
                    double vapour_pressure = (
                            33.864 * (pow(7.38e-3*T + 0.8072, 8) -
                                1.9e-5*fabs(1.8*T + 48.0) + 1.316e-3) *
                            rel_hum);
                    if (T < 0.0)  // Change VP over water to VP over ice (Bosen)
                        vapour_pressure *= 1.0 + 9.72e-3*T + 4.2e-5*T*T;

                    // There is some snowfall. Here we update the snowpack and
                    // wet snow to reflect the new snow. We also
                    // update albedo.
                    if (snow > hbv_physical_snow::tol) {
                        auto idx = sca_index(sca);
                        if (sca > 1.0e-5 && sca < 1.0 - 1.0e-5) {
                            if (idx == 0) {
                                sp[0] *= sca/(I[1] - I[0]);
                                sw[0] *= sca/(I[1] - I[0]);
                            } else {
                                sp[idx] *= (1.0 + (sca - I[idx]) / (I[idx] -
                                            I[idx - 1])) / (1.0 + (I[idx + 1] -
                                                I[idx])/(I[idx] - I[idx - 1]));
                                sw[idx] *= (1.0 + (sca - I[idx]) / (I[idx] -
                                            I[idx - 1])) / (1.0 + (I[idx + 1] -
                                                I[idx])/(I[idx] - I[idx - 1]));
                            }
                        }

                        for (size_t i = 0; i < I_n; ++i)
                        {
                            double currsnow = snow * sd[i];
                            sp[i] += currsnow;
                            albedo[i] += (currsnow * albedo_range /
                                          p.snowfall_reset_depth);
                        }

                        for (size_t i = I_n - 2; i > 0; --i)
                            if (sd[i] > 0.0) {
                                sca = sd[i + 1];
                                break;
                            } else
                                sca = sd[1];
                    } else {
                        // No snowfall: Albedo decays
                        if (T < 0.0) {
                            for (auto &alb: albedo) {
                                alb -= slow_albedo_decay_rate;
                            }
                        } else {
                            for (auto &alb: albedo) {
                                alb = (min_albedo + fast_albedo_decay_rate *
                                       (alb - min_albedo));
                            }
                        }
                    }

                    // We now start calculation of energy content based on
                    // current albedoes etc.

                    for (auto &alb: albedo) {
                        alb = std::max(std::min(alb, max_albedo),
                                       min_albedo);
                    }

                    vector<double> effect;
                    for (auto alb: albedo) {
                        effect.push_back(rad * (1.0 - alb));
                    }


                    for (auto &eff: effect) {
                        eff += (0.98 * sigma *
                                pow(vapour_pressure/T_k, 6.87e-2) *
                                pow(T_k, 4));
                    }

                    if (T > 0.0 && snow < hbv_physical_snow::tol) {
                        for (auto &eff: effect) {
                            eff += rain * T * water_heat/(double)dt;
                        }
                    }
                    if (T <= 0.0 && rain < hbv_physical_snow::tol)
                        for (size_t i=0; i<I_n; ++i) {
                            effect[i] += snow*sd[i]*T*ice_heat/(double)dt;
                        }
                    //TODO: Should the snow distribution be included here?
                //        effect += snow*T*ice_heat/(double)dt;

                    if (p.calculate_iso_pot_energy) {
                        for (size_t i=0; i<I_n; ++i) {
                            double iso_effect = (effect[i] - BB0 + turb *
                                    (T + 1.7 * (vapour_pressure - 6.12)));
                            iso_pot_energy[i] += (iso_effect *
                                    (double)dt/melt_heat);
                        }
                    }

                    double sst = std::min(0.0, 1.16*T - 2.09);
                    if (sst > -hbv_physical_snow::tol) {
                        for (auto &eff: effect) {
                            eff += turb * (T + 1.7 *
                                    (vapour_pressure - 6.12)) - BB0;
                        }
                    } else {
                        for (auto &eff: effect) {
                            eff += (turb * (T - sst + 1.7 *
                                        (vapour_pressure - 6.132 *
                                         exp(0.103 * T - 0.186))) -
                                    0.98 * sigma * pow(sst + 273.15, 4));
                        }
                    }

                    // Surface heat change, positive during warming
                    double delta_sh = -surface_heat;

                    // New surface heat; always nonpositive since sst <= 0.
                    surface_heat = p.surface_magnitude*ice_heat*sst*0.5;
                    delta_sh += surface_heat;

                    vector<double> energy;
                    for (auto eff : effect) {
                        energy.push_back(eff * (double)dt);
                    }
                    if (delta_sh > 0.0) {
                        //TODO: Should this condition be removed when we allow refreeze?
                //        energy -= delta_sh;  // Surface energy is a sink, but not a source
                        for (auto &en: energy) en -= delta_sh;
                    }

                    vector<double> potential_melt;
                    for (auto en: energy) {
                        //This is a difference from the physical model: We
                        //allow negative potential melts. This to fit it better
                        //with the HBV model. We treat negative potential melts
                        //as refreeze.
                        potential_melt.push_back(en/melt_heat);
                    }

                    // We have now calculated the potential melt in each bin
                    // and now update the distributions and outflow etc. to
                    // reflect that.
                    const double lw = p.lw;

                    size_t idx = I_n;
                    bool any_melt = false;

                    for (size_t i=0; i<I_n; ++i) {
                        if (potential_melt[i] >= hbv_physical_snow::tol) {
                            any_melt = true;
                            if (sp[i] < potential_melt[i]) {
                                idx = i;
                                break;
                            }
                        }
                    }

                    // If there is melting at all
                    if (any_melt) {
                        if (idx == 0) sca = 0.0;
                        else if (idx == I_n) sca = 1.0;
                        else {
                            if (sp[idx] > 0.0) {
                                sca = (I[idx] - (I[idx] - I[idx - 1]) *
                                        (potential_melt[idx] - sp[idx]) /
                                        (sp[idx - 1] = sp[idx]));
                            } else {
                                sca = (1.0 - potential_melt[idx]/sp[idx - 1]) *
                                    (sca - I[idx - 1]) + I[idx - 1];
                            }
                        }
                    }


                    // If negative melt, we treat it as refreeze,
                    // otherwise we update the snowpack and wet snow.
                    for (size_t i=0; i<I_n; ++i) {
                        if (potential_melt[i] < hbv_physical_snow::tol) {
                            refreeze(sp[i], sw[i], rain,
                                     p.cfr*potential_melt[i], lw);
                        } else {
                            update_state(sp[i], sw[i], rain, potential_melt[i],
                                         lw);
                        }
                    }

                    if (sca < hbv_physical_snow::tol) swe = 0.0;
                    else {
                        bool f_is_zero = sca >= 1.0 ? false : true;
                        swe = integrate(sp, I, I_n, 0, sca, f_is_zero);
                        swe += integrate(sw, I, I_n, 0, sca, f_is_zero);
                    }

                    if (total_water < swe) {
                        if (total_water - swe < -hbv_physical_snow::tol) {
                            ostringstream buff;
                            buff << "Negative outflow: total_water (" <<
                                total_water << ") - swe (" << swe << ") = "
                                << total_water - swe;
                            throw runtime_error(buff.str());
                        } else
                            swe = total_water;
                    }
                    r.outflow = total_water - swe;
                    r.sca = sca;
                    r.storage = swe;
                    s.swe = swe;
                    s.sca = sca;
                    s.iso_pot_energy = iso_pot_energy;
                }
            };
        }
    }
}

x_serialize_export_key(shyft::core::hbv_physical_snow::state);
