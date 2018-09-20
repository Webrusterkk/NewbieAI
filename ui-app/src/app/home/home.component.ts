import { Component, OnInit } from "@angular/core";
import { FormBuilder, FormGroup, Validators, FormControl } from "@angular/forms";
import { Constants } from "../common/constants";
import { Observable } from 'rxjs';
import { map, startWith } from 'rxjs/operators';
import { SimpleObject } from "../models/simple-object";
import { HomeService } from "./home.service";

@Component({
    selector: "home",
    templateUrl: "./home.component.html",
    styleUrls: ["./home.component.css"],
    providers: [HomeService]
})
export class HomeComponent implements OnInit {
    title: string;
    public homeForm: FormGroup;
    vesselTypes: string[] = Constants.VesselTypes;
    vesselSubTypes: string[] = Constants.VesselSubTypes;
    ports: string[] = Constants.Prots;
    // types: SimpleObject[] = Constants.VesselTypes;
    constructor(private formBuilder: FormBuilder,
        private _homeService: HomeService) {
    }

    ngOnInit() {
        this.title = "Home - Newbie";
        this.homeForm = this.formBuilder.group({
            vesselType: [null, Validators.required],
            vesselSubType: [null, Validators.required],
            vesselAge: [null, Validators.required],
            sourceCountry: [null, Validators.required],
            destinationCountry: [null, Validators.required],
            date: [null, Validators.required]
        })
    }

    onSubmit() {
        console.log(this.homeForm.value);
    }

}