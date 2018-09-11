import { Component, OnInit } from "@angular/core";
import { FormBuilder, FormGroup, Validators, FormControl } from "@angular/forms";

@Component({
    selector: "home",
    templateUrl: "./home.component.html",
    styleUrls: ["./home.component.css"],
    providers:[FormBuilder]
})
export class HomeComponent implements OnInit {
    title: string;
    public homeForm: FormGroup;
    constructor(private formBuilder: FormBuilder) {
    }

    ngOnInit() {
        this.title = "Home - Newbie";

        this.homeForm = this.formBuilder.group({
            vesselType: [0],
            vesselAge: [""],
            sourcePort: [""],
            destinationPort: [""],
            date: []
        })
    }
}