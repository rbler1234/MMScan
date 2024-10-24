# Some templates of the MMScan samples
MMScan_QA_template = { "sub_class": "QA_Single_EQ",
                        "scan_id": "1mp3d_0000_region1",
                        "question": "Look carefully at the room, is there a shelf in the room? ",
                        "answers": [
                            "Yes, there is."
                        ],
                        "object_ids": [
                            1
                        ],
                        "object_names": [
                            "shelf"
                        ],
                        "input_bboxes_id": None,
                        "input_bboxes": None,
                        "output_bboxes_id": None,
                        "output_bboxes": None,
                        "ID": "QA_Single_EQ__1mp3d_0000_region1__1"
                    },


MMScan_VG_template = {
                        "sub_class": "VG_Single_Attribute_Common",
                        "scan_id": "1mp3d_0000_region2",
                        "target_id": [
                            4,
                            8
                        ],
                        "distractor_ids": [],
                        "text": "Look carefully at the room, find all basic components and accessories of the building. ",
                        "target": [
                            "doorframe",
                            "door"
                        ],
                        "anchors": [],
                        "anchor_ids": [],
                        "tokens_positive": {
                            "4": [
                                [
                                    33,
                                    85
                                ]
                            ],
                            "8": [
                                [
                                    33,
                                    85
                                ]
                            ]
                        },
                        "ID": "VG_Single_Attribute_Common__1mp3d_0000_region2__1"
                    }