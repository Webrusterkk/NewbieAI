import { Component, OnInit } from "@angular/core";
import { FormBuilder, FormGroup, Validators, FormControl } from "@angular/forms";
import { Constants } from "../common/constants";
import { Observable } from 'rxjs';
import { map, startWith } from 'rxjs/operators';
import { SimpleObject } from "../models/simple-object";

@Component({
    selector: "home",
    templateUrl: "./home.component.html",
    styleUrls: ["./home.component.css"],
    providers: [FormBuilder]
})
export class HomeComponent implements OnInit {
    title: string;
    public homeForm: FormGroup;
    vesselTypes: string[] = Constants.VesselTypes;
    // types: SimpleObject[] = Constants.VesselTypes;
    constructor(private formBuilder: FormBuilder) {
    }

    ngOnInit() {
        this.title = "Home - Newbie";
        this.homeForm = this.formBuilder.group({
            vesselType: [null, Validators.required],
            vesselAge: [null, Validators.required],
            sourcePort: [null, Validators.required],
            destinationPort: [null, Validators.required],
            date: [null, Validators.required]
        })
    }

    onSubmit(){
        console.log(this.homeForm.value);
    }

}