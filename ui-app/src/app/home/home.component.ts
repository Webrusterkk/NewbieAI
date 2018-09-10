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
    public options: FormGroup;
    age : FormControl;
    constructor(private formBuilder: FormBuilder) {
    }

    ngOnInit() {
        this.title = "Home - Newbie";

        this.options = this.formBuilder.group({
            vesselType: [0],
            vesselAge: [""],
            sourcePort: [""],
            destinationPort: [""]
        })

        this.age = new FormControl();
    }
}