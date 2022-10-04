# GNN State Estimation


## NetworkX Data Sample
```json
{
  "directed": false,
  "multigraph": false,
  "graph": {
    "targets": [
      "voltage"
    ],
    "task": "node_regression"
  },
  "nodes": [
    {
      "nodetype": "Node",
      "id": 0
    },
    {
      "nodetype": "Line",
      "m_fltBaseNomKV": 11.0,
      "m_fltBch_pu": 9.99e-12,
      "m_fltGch_pu": 9.99,
      "m_fltR1_pu": 0.00099999,
      "m_fltX1_pu": 0.00099999,
      "id": 107
    },
    {
      "nodetype": "Line",
      "m_fltBaseNomKV": 11.0,
      "m_fltBch_pu": 0.00099999,
      "m_fltGch_pu": 0.00099999,
      "m_fltR1_pu": 0.00099999,
      "m_fltX1_pu": 0.00099999,
      "cMvI": 999.9999,
      "cMvI_side": "cBrSideB",
      "measured_values": [
        [
           999.9999,
          "PhaseA",
          "cBrSideB",
          "cMvI",
          "1999-10-23T09:51:55.537Z"
        ],
        [
          999.9999,
          "PhaseB",
          "cBrSideB",
          "cMvI",
          "1999-10-23T09:51:55.537Z"
        ],
        [
           999.9999,
          "PhaseC",
          "cBrSideB",
          "cMvI",
          "1999-10-23T09:51:55.537Z"
        ]
      ],
      "id": 108
    },
    {
      "nodetype": "Trafo",
      "id": 147
    },
    {
      "nodetype": "Winding",
      "m_fltBaseNomKV": 11.0,
      "m_fltPT_Ratio": 1.0,
      "m_enumConnectionType": "cDelta",
      "m_enumWindingType": "cPrimary",
      "m_fltBmag_pu":  999.9999,
      "m_fltGmag_pu":  999.9999,
      "m_fltR1_pu":  999.9999,
      "m_fltRatedkV": 11.0,
      "m_fltRatedkVA":  999.9999,
      "m_fltX1_pu":  999.9999,
      "m_sVectorGroup": 0,
      "id": 148
    },
    {
      "nodetype": "Load",
      "m_fltBaseNomKV": 11.0,
      "m_fltP_kW":  999.9999,
      "m_fltQ_kVAr":  999.9999,
      "P":  999.9999,
      "Q":  999.9999,
      "cos_Phi": 0,
      "calc_I":  999.9999,
      "Phase": 7,
      "ConnectionType": 2,
      "Value_S_real_pu_A":  999.9999,
      "Value_S_imag_pu_A":  999.9999,
      "Value_S_real_kVA_A":  999.9999,
      "Value_S_imag_kVA_A":  999.9999,
      "Value_I_real_pu_A":  999.9999,
      "Value_I_imag_pu_A":  999.9999,
      "Value_I_real_Amps_A":  999.9999,
      "Value_I_imag_Amps_A": 999.9999,
      "PhaseImbalance_I":  999.9999,
      "Value_V_real_pu_A":  999.9999,
      "Value_V_imag_pu_A":  999.9999,
      "Value_V_LL_kV_A":  999.9999,
      "Value_V_LN_kV_A":  999.9999,
      "Value_V_angle_degrees_A":  999.9999,
      "Value_PowerFactor_A":  999.9999,
      "LoadToVoltageDependencyCoefficient_real_A": 1.0,
      "LoadToVoltageDependencyCoefficient_imag_A": 1.0,
      "LoadSource": 0,
      "cMvI":  999.9999,
      "cMvI_side": "cNotBranch",
      "measured_values": [
        [
           999.9999,
          "PhaseA",
          "cNotBranch",
          "cMvI",
          "1999-10-23T09:51:55.537Z"
        ]
      ],
      "id": 184
    },
    {
      "nodetype": "Bb",
      "m_fltBaseNomKV":  999.9999,
      "P":  999.9999,
      "Q":  999.9999,
      "cos_Phi": 0,
      "calc_I":  999.9999,
      "cMvCosPhi":  999.9999,
      "cMvCosPhi_side": "cNotBranch",
      "cMvV":  999.9999,
      "cMvV_side": "cNotBranch",
      "measured_values": [
        [
          0,
          "PhaseAB",
          "cNotBranch",
          "cMvCosPhi",
          "1999-10-23T09:51:55.537Z"
        ],
        [
           999.9999,
          "PhaseAB",
          "cNotBranch",
          "cMvV",
          "1999-10-23T09:51:55.537Z"
        ]
      ],
      "voltage":  999.9999,
      "id": 215
    }
  ],
  "links": [
    {
      "side": "cBrSideB",
      "source": 0,
      "target": 107
    },
    {
      "source": 108,
      "target": 215
    }
    ...
  ]
}

```
